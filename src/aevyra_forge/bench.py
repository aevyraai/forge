# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Benchmark verifier — replays a workload against a running vLLM server.

See AGENT.md → "Module-by-module spec → bench.py".

Output is a structured ``BenchResult``. The orchestrator's score
function combines these fields into a single comparable number,
subject to the accuracy floor in ``ForgeConfig``.

Requests are dispatched concurrently up to ``workload.concurrency``
in-flight at once using ``asyncio`` + ``httpx.AsyncClient``.  This puts
real batch pressure on vLLM so that ``max_num_seqs``,
``max_num_batched_tokens``, and chunked-prefill knobs are exercised.

Streaming (``stream=True`` + ``stream_options.include_usage``) is used
so that TTFT is measured from the first SSE chunk and token counts come
from the server-reported usage in the final chunk.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from aevyra_forge.workload import Workload, WorkloadRequest

if TYPE_CHECKING:
    import httpx


logger = logging.getLogger(__name__)

BenchStatus = Literal["PASS", "FAIL", "TIMEOUT", "CRASH"]


@dataclass
class BenchResult:
    # --- Performance ---
    throughput_tokens_per_sec: float
    p50_latency_ms: float
    p99_latency_ms: float
    ttft_ms: float
    tpot_ms: float

    # --- Capacity ---
    peak_vram_mb: int
    max_concurrent_seqs: int

    # --- Accuracy ---
    accuracy_score: float | None
    accuracy_delta_vs_baseline: float | None

    # --- Provenance / status ---
    recipe_id: str
    workload_id: str
    bench_duration_s: float
    status: BenchStatus
    error: str | None = None


# ---------------------------------------------------------------------------
# Async core
# ---------------------------------------------------------------------------


async def _send_one(
    client: "httpx.AsyncClient",  # type: ignore[name-defined]
    req: WorkloadRequest,
    *,
    model_id: str,
    use_chat: bool,
    semaphore: asyncio.Semaphore,
    timeout_s: int,
) -> tuple[float, float, int, str | None]:
    """Send one request and return (latency_ms, ttft_ms, n_output_tokens, error).

    Uses streaming so TTFT is measured at the first SSE chunk and token
    counts come from the server-side ``usage`` field in the final chunk.
    """
    import time

    async with semaphore:
        t0 = time.monotonic()
        ttft_ms = 0.0
        n_tokens = 0
        first_chunk = True

        if use_chat:
            payload: dict = {
                "model": model_id,
                "messages": [{"role": "user", "content": req.prompt}],
                "max_tokens": req.expected_output_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            endpoint = "/v1/chat/completions"
        else:
            payload = {
                "model": model_id,
                "prompt": req.prompt,
                "max_tokens": req.expected_output_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            endpoint = "/v1/completions"

        try:
            async with client.stream(
                "POST", endpoint, json=payload, timeout=float(timeout_s)
            ) as resp:
                if resp.status_code != 200:
                    return 0.0, 0.0, 0, f"HTTP {resp.status_code}"

                async for raw_line in resp.aiter_lines():
                    if not raw_line.startswith("data:"):
                        continue
                    data_str = raw_line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break

                    if first_chunk:
                        ttft_ms = (time.monotonic() - t0) * 1000
                        first_chunk = False

                    try:
                        chunk = _json.loads(data_str)
                    except _json.JSONDecodeError:
                        continue

                    # Pick up usage from the final chunk (stream_options)
                    usage = chunk.get("usage")
                    if usage:
                        n_tokens = usage.get("completion_tokens", n_tokens)

            latency_ms = (time.monotonic() - t0) * 1000
            # Fallback: if server didn't return usage, use expected token count
            if n_tokens == 0:
                n_tokens = req.expected_output_tokens
            return latency_ms, ttft_ms, n_tokens, None

        except asyncio.TimeoutError:
            return 0.0, 0.0, 0, "timeout"
        except Exception as exc:
            return 0.0, 0.0, 0, str(exc)


async def _benchmark_async(
    *,
    server_url: str,
    workload: Workload,
    model_id: str,
    use_chat: bool,
    concurrency: int,
    timeout_s: int,
) -> tuple[list[float], list[float], int, list[str]]:
    """Dispatch all workload requests with bounded concurrency.

    Returns (latencies_ms, ttfts_ms, total_output_tokens, errors).
    """
    try:
        import httpx
    except ImportError as e:
        raise ImportError("bench requires httpx: pip install aevyra-forge[vllm]") from e

    semaphore = asyncio.Semaphore(concurrency)

    # Progress bar — tqdm.asyncio if available
    try:
        from tqdm.asyncio import tqdm as _atqdm

        async def _gather(coros):
            return await _atqdm.gather(
                *coros,
                total=len(workload.requests),
                desc=f"forge │  bench (c={concurrency})",
                unit="req",
                dynamic_ncols=True,
            )

    except ImportError:

        async def _gather(coros):  # type: ignore[misc]
            return await asyncio.gather(*coros)

    async with httpx.AsyncClient(base_url=server_url) as client:
        results = await _gather(
            [
                _send_one(
                    client,
                    req,
                    model_id=model_id,
                    use_chat=use_chat,
                    semaphore=semaphore,
                    timeout_s=timeout_s,
                )
                for req in workload.requests
            ]
        )

    latencies: list[float] = []
    ttfts: list[float] = []
    total_output_tokens = 0
    errors: list[str] = []

    for latency_ms, ttft_ms, n_tok, err in results:
        if err is not None:
            errors.append(err)
        else:
            latencies.append(latency_ms)
            ttfts.append(ttft_ms)
            total_output_tokens += n_tok

    return latencies, ttfts, total_output_tokens, errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def warmup(
    *,
    server_url: str,
    workload: Workload,
    n_requests: int = 10,
    timeout_s: int = 120,
) -> None:
    """Send a small subset of the workload to bring vLLM to steady state.

    Ensures every experiment is benchmarked with a warm KV prefix cache and
    warmed-up CUDA kernels, making comparisons fair regardless of whether
    vLLM was just restarted or reused from the previous experiment.

    Uses the first ``n_requests`` which share the common prefix (if any) so
    that the prefix cache is populated before the timed run starts. Errors
    are silently ignored — this is best-effort.
    """
    if not workload.requests:
        return
    warm_requests = workload.requests[:n_requests]
    warm_wl = Workload(
        requests=warm_requests,
        duration_s=workload.duration_s,
        concurrency=min(workload.concurrency, n_requests),
    )
    logger.info("forge │  warmup: %d requests (cache prime + kernel warm)", len(warm_requests))
    try:
        benchmark(
            server_url=server_url,
            workload=warm_wl,
            recipe_id="warmup",
            workload_id="warmup",
            accuracy_check=False,
            timeout_s=timeout_s,
        )
    except Exception as exc:
        logger.debug("Warmup failed (non-fatal): %s", exc)


def benchmark(
    *,
    server_url: str,
    workload: Workload,
    recipe_id: str,
    workload_id: str,
    accuracy_check: bool = True,
    timeout_s: int = 600,
    dry_run: bool = False,
) -> BenchResult:
    """Replay ``workload`` against the server at ``server_url``.

    Concurrency is taken from ``workload.concurrency`` (default 1 →
    sequential, identical to the old behaviour).  Set it to 8–32 via
    :func:`~aevyra_forge.workload.workload_concurrent_synthetic` or by
    setting the field directly on your workload object.

    Never raises — failures are captured in ``status`` and ``error``.
    """
    import time

    if dry_run:
        import random

        rng = random.Random(recipe_id)
        return BenchResult(
            throughput_tokens_per_sec=rng.uniform(800, 1200),
            p50_latency_ms=rng.uniform(40, 80),
            p99_latency_ms=rng.uniform(100, 300),
            ttft_ms=rng.uniform(20, 60),
            tpot_ms=rng.uniform(5, 15),
            peak_vram_mb=rng.randint(16000, 40000),
            max_concurrent_seqs=128,
            accuracy_score=None,
            accuracy_delta_vs_baseline=None,
            recipe_id=recipe_id,
            workload_id=workload_id,
            bench_duration_s=rng.uniform(10, 30),
            status="PASS",
        )

    try:
        import httpx
    except ImportError as e:
        raise ImportError("bench requires httpx: pip install aevyra-forge[vllm]") from e

    # Probe which endpoint and model ID to use
    with httpx.Client(base_url=server_url, timeout=30) as probe:
        try:
            _r = probe.get("/v1/models", timeout=5.0)
            _models = _r.json().get("data", [])
            model_id = _models[0]["id"] if _models else "default"
        except Exception:
            model_id = "default"

        _chat_probe = probe.post(
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
            timeout=10.0,
        )
        use_chat = _chat_probe.status_code == 200

    concurrency = max(1, workload.concurrency)
    logger.info(
        "forge │  bench: %d requests  concurrency=%d  endpoint=%s",
        len(workload.requests),
        concurrency,
        "/v1/chat/completions" if use_chat else "/v1/completions",
    )

    t_start = time.monotonic()

    _coro = _benchmark_async(
        server_url=server_url,
        workload=workload,
        model_id=model_id,
        use_chat=use_chat,
        concurrency=concurrency,
        timeout_s=timeout_s,
    )

    try:
        # Colab / Jupyter kernels run their own event loop, so asyncio.run()
        # raises "This event loop is already running". Work around by running
        # the coroutine in a dedicated background thread that owns its loop.
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None and running_loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _coro)
                latencies, ttfts, total_output_tokens, errors = future.result(
                    timeout=timeout_s + 30
                )
        else:
            latencies, ttfts, total_output_tokens, errors = asyncio.run(_coro)
    except Exception as exc:
        _coro.close()  # prevent "coroutine never awaited" RuntimeWarning
        elapsed = time.monotonic() - t_start
        return BenchResult(
            throughput_tokens_per_sec=0.0,
            p50_latency_ms=0.0,
            p99_latency_ms=0.0,
            ttft_ms=0.0,
            tpot_ms=0.0,
            peak_vram_mb=0,
            max_concurrent_seqs=0,
            accuracy_score=None,
            accuracy_delta_vs_baseline=None,
            recipe_id=recipe_id,
            workload_id=workload_id,
            bench_duration_s=elapsed,
            status="CRASH",
            error=str(exc),
        )

    elapsed = time.monotonic() - t_start

    if not latencies:
        return BenchResult(
            throughput_tokens_per_sec=0.0,
            p50_latency_ms=0.0,
            p99_latency_ms=0.0,
            ttft_ms=0.0,
            tpot_ms=0.0,
            peak_vram_mb=0,
            max_concurrent_seqs=0,
            accuracy_score=None,
            accuracy_delta_vs_baseline=None,
            recipe_id=recipe_id,
            workload_id=workload_id,
            bench_duration_s=elapsed,
            status="FAIL",
            error="; ".join(errors[:5]),
        )

    latencies.sort()
    ttfts_sorted = sorted(t for t in ttfts if t > 0)

    def _pct(lst: list[float], p: float) -> float:
        if not lst:
            return 0.0
        idx = max(0, int(len(lst) * p / 100) - 1)
        return lst[idx]

    throughput = total_output_tokens / elapsed if elapsed > 0 else 0.0
    avg_latency = sum(latencies) / len(latencies)
    avg_output = total_output_tokens / len(latencies) if latencies else 1
    tpot = avg_latency / avg_output if avg_output else 0.0

    return BenchResult(
        throughput_tokens_per_sec=throughput,
        p50_latency_ms=_pct(latencies, 50),
        p99_latency_ms=_pct(latencies, 99),
        ttft_ms=_pct(ttfts_sorted, 50) if ttfts_sorted else 0.0,
        tpot_ms=tpot,
        peak_vram_mb=0,
        max_concurrent_seqs=concurrency,
        accuracy_score=None,
        accuracy_delta_vs_baseline=None,
        recipe_id=recipe_id,
        workload_id=workload_id,
        bench_duration_s=elapsed,
        status="PASS" if not errors else "FAIL",
        error="; ".join(errors[:5]) if errors else None,
    )


def score(
    result: BenchResult,
    *,
    accuracy_floor: float = 0.99,
) -> float:
    """Combine the bench result into a single number the orchestrator can compare.

    v0 score = throughput, gated on accuracy >= floor.
    """
    if result.status != "PASS":
        return 0.0
    if result.accuracy_delta_vs_baseline is not None and result.accuracy_delta_vs_baseline < (
        accuracy_floor - 1.0
    ):
        return 0.0
    return result.throughput_tokens_per_sec

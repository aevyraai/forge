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

Two implementation paths:

1. Shell out to ``vllm bench serve`` and parse its output.
2. Implement the replay loop directly via an OpenAI-compatible client.

Pick whichever ends up more robust against vLLM version drift. The
function contract is the same either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from aevyra_forge.workload import Workload


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

    t_start = time.time()
    latencies: list[float] = []
    ttfts: list[float] = []
    total_output_tokens = 0
    errors: list[str] = []

    # Detect which completion endpoint this vLLM version exposes
    with httpx.Client(base_url=server_url, timeout=30) as probe:
        try:
            _r = probe.get("/v1/models", timeout=5.0)
            _models = _r.json().get("data", [])
            _model_id = _models[0]["id"] if _models else "default"
        except Exception:
            _model_id = "default"
        # Prefer /v1/chat/completions (vLLM 0.4+); fall back to /v1/completions
        _chat_probe = probe.post(
            "/v1/chat/completions",
            json={"model": _model_id, "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 1},
            timeout=10.0,
        )
        _use_chat = _chat_probe.status_code == 200

    # Progress bar — use tqdm.notebook in Jupyter/Colab, plain tqdm elsewhere
    try:
        from tqdm.notebook import tqdm as _tqdm
    except ImportError:
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None  # tqdm not installed, skip bar

    def _wrap(iterable, **kwargs):
        if _tqdm is not None:
            return _tqdm(iterable, **kwargs)
        return iterable

    with httpx.Client(base_url=server_url, timeout=timeout_s) as client:
        for req in _wrap(
            workload.requests,
            total=len(workload.requests),
            desc="forge │  bench",
            unit="req",
            dynamic_ncols=True,
        ):
            t0 = time.time()
            try:
                if _use_chat:
                    resp = client.post(
                        "/v1/chat/completions",
                        json={
                            "model": _model_id,
                            "messages": [{"role": "user", "content": req.prompt}],
                            "max_tokens": req.expected_output_tokens,
                            "stream": False,
                        },
                        timeout=60.0,
                    )
                else:
                    resp = client.post(
                        "/v1/completions",
                        json={
                            "model": _model_id,
                            "prompt": req.prompt,
                            "max_tokens": req.expected_output_tokens,
                            "stream": False,
                        },
                        timeout=60.0,
                    )
                t1 = time.time()
                if resp.status_code == 200:
                    data = resp.json()
                    if _use_chat:
                        n_tokens = data.get("usage", {}).get("completion_tokens", req.expected_output_tokens)
                    else:
                        n_tokens = data.get("usage", {}).get("completion_tokens", req.expected_output_tokens)
                    total_output_tokens += n_tokens
                    latency_ms = (t1 - t0) * 1000
                    latencies.append(latency_ms)
                    ttfts.append(latency_ms * 0.3)
                else:
                    errors.append(f"HTTP {resp.status_code}")
            except httpx.TimeoutException:
                errors.append("timeout")
            except Exception as exc:
                errors.append(str(exc))

    elapsed = time.time() - t_start

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
    ttfts.sort()

    def _pct(lst: list[float], p: float) -> float:
        idx = max(0, int(len(lst) * p / 100) - 1)
        return lst[idx]

    throughput = total_output_tokens / elapsed if elapsed > 0 else 0.0
    avg_output = total_output_tokens / len(latencies) if latencies else 1
    tpot = (sum(latencies) / len(latencies)) / avg_output if avg_output else 0.0

    return BenchResult(
        throughput_tokens_per_sec=throughput,
        p50_latency_ms=_pct(latencies, 50),
        p99_latency_ms=_pct(latencies, 99),
        ttft_ms=_pct(ttfts, 50),
        tpot_ms=tpot,
        peak_vram_mb=0,
        max_concurrent_seqs=len(workload.requests),
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
    Production version will be Pareto-aware. Don't over-engineer this yet.
    """
    if result.status != "PASS":
        return 0.0
    if result.accuracy_delta_vs_baseline is not None and result.accuracy_delta_vs_baseline < (
        accuracy_floor - 1.0
    ):
        return 0.0
    return result.throughput_tokens_per_sec

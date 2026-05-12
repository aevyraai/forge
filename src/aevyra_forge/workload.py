# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Workload trace — input to the benchmark verifier.

See AGENT.md → "Key concepts → The workload" and
"Module-by-module spec → workload.py".

Sources, in order of preference:

1. Production trace (Langfuse / OTel / JSONL) — the customer's actual traffic.
2. Synthetic shape match — when only distribution summaries are available.
3. ShareGPT / public — fallback for the Colab demo.

Workloads are replayed deterministically: same seed → same arrival timing.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WorkloadRequest:
    prompt: str
    expected_output_tokens: int
    arrival_offset_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Workload:
    requests: list[WorkloadRequest]
    duration_s: float
    metadata: dict[str, Any] = field(default_factory=dict)
    concurrency: int = 1  # max simultaneous in-flight requests during benchmarking

    def summary(self) -> dict[str, Any]:
        """Distributions the agent prompt can read."""
        if not self.requests:
            return {
                "request_count": 0,
                "duration_s": self.duration_s,
                "input_token_p50": 0,
                "input_token_p99": 0,
                "output_token_p50": 0,
                "output_token_p99": 0,
                "requests_per_sec": 0.0,
                "prefix_cache_hit_estimate": 0.0,
            }

        # Estimate input tokens as whitespace-split word count * 1.3
        input_lens = sorted(int(len(r.prompt.split()) * 1.3) for r in self.requests)
        output_lens = sorted(r.expected_output_tokens for r in self.requests)

        def pct(lst: list[int], p: float) -> int:
            idx = max(0, int(len(lst) * p / 100) - 1)
            return lst[idx]

        # Rough prefix cache estimate: fraction of requests sharing a common prefix
        # (heuristic: count requests whose first 20 tokens match most others)
        prefix_hit = 0.0
        if len(self.requests) > 1:
            prefix_samples = [r.prompt[:80] for r in self.requests[:200]]
            common = max(set(prefix_samples), key=prefix_samples.count)
            prefix_hit = round(prefix_samples.count(common) / len(prefix_samples), 2)

        rps = len(self.requests) / self.duration_s if self.duration_s > 0 else 0.0

        return {
            "request_count": len(self.requests),
            "duration_s": self.duration_s,
            "input_token_p50": pct(input_lens, 50),
            "input_token_p99": pct(input_lens, 99),
            "output_token_p50": pct(output_lens, 50),
            "output_token_p99": pct(output_lens, 99),
            "requests_per_sec": round(rps, 2),
            "prefix_cache_hit_estimate": prefix_hit,
        }


def workload_from_jsonl(path: str | Path) -> Workload:
    """Load a workload from a JSONL file.

    Expected line format::

        {"prompt": "...", "expected_output_tokens": 128, "arrival_offset_s": 0.0}
    """
    path = Path(path)
    requests: list[WorkloadRequest] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        requests.append(
            WorkloadRequest(
                prompt=d["prompt"],
                expected_output_tokens=int(d.get("expected_output_tokens", 128)),
                arrival_offset_s=float(d.get("arrival_offset_s", 0.0)),
                metadata={
                    k: v
                    for k, v in d.items()
                    if k not in ("prompt", "expected_output_tokens", "arrival_offset_s")
                },
            )
        )
    duration_s = requests[-1].arrival_offset_s + 1.0 if requests else 1.0
    return Workload(requests=requests, duration_s=duration_s)


def workload_concurrent_synthetic(
    *,
    n_requests: int = 200,
    concurrency: int = 16,
    avg_input_tokens: int = 256,
    avg_output_tokens: int = 128,
    duration_s: float = 60.0,
    seed: int = 0,
) -> Workload:
    """Synthetic workload designed to stress vLLM's batching engine.

    Unlike :func:`workload_synthetic`, all requests are sent concurrently
    (up to *concurrency* in-flight at once) so that ``max_num_seqs``,
    ``max_num_batched_tokens``, and ``enable_chunked_prefill`` are actually
    exercised.  Use this instead of the default sequential workload whenever
    you want config-layer mutations to have a measurable effect.

    Typical concurrency values:
    - T4 / A10 (16–24 GB):  concurrency=8–16
    - A100 / H100 (80 GB):  concurrency=32–64
    """
    wl = workload_synthetic(
        n_requests=n_requests,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        duration_s=duration_s,
        seed=seed,
    )
    wl.concurrency = concurrency
    wl.metadata["id"] = f"concurrent-synthetic-c{concurrency}"
    return wl


def workload_from_langfuse(path: str | Path) -> Workload:
    """Load a Langfuse export. v0: thin wrapper around workload_from_jsonl."""
    return workload_from_jsonl(path)


def workload_from_otel(spans: list[dict[str, Any]]) -> Workload:
    """Convert OTel GenAI spans into a Workload. v0: stub."""
    raise NotImplementedError("workload_from_otel is v0.2+")


def workload_synthetic(
    *,
    n_requests: int = 1000,
    avg_input_tokens: int = 512,
    avg_output_tokens: int = 128,
    duration_s: float = 60.0,
    seed: int = 0,
) -> Workload:
    """Generate a ShareGPT-shaped synthetic workload for demos / smoke tests."""
    rng = random.Random(seed)

    # Lognormal parameters: mu/sigma s.t. mean ≈ avg
    def _lognormal_sample(avg: int) -> int:
        sigma = 0.5
        mu = math.log(avg) - 0.5 * sigma**2
        return max(1, int(rng.lognormvariate(mu, sigma)))

    # Poisson arrival times
    arrival = 0.0
    rate = n_requests / duration_s  # requests per second

    requests: list[WorkloadRequest] = []
    for i in range(n_requests):
        inter_arrival = rng.expovariate(rate)
        arrival += inter_arrival
        prompt_tokens = _lognormal_sample(avg_input_tokens)
        output_tokens = _lognormal_sample(avg_output_tokens)
        # Build a plausible-length dummy prompt
        prompt = " ".join(f"word{rng.randint(0, 9999)}" for _ in range(max(1, prompt_tokens // 2)))
        requests.append(
            WorkloadRequest(
                prompt=prompt,
                expected_output_tokens=output_tokens,
                arrival_offset_s=round(arrival, 4),
            )
        )

    return Workload(requests=requests, duration_s=duration_s)

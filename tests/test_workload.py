# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for aevyra_forge.workload."""

import json
from pathlib import Path

import pytest

from aevyra_forge.workload import (
    Workload,
    WorkloadRequest,
    workload_concurrent_synthetic,
    workload_from_jsonl,
    workload_shared_prefix,
    workload_synthetic,
)


# ---------------------------------------------------------------------------
# WorkloadRequest
# ---------------------------------------------------------------------------


def test_workload_request_defaults() -> None:
    req = WorkloadRequest(prompt="hello world", expected_output_tokens=64, arrival_offset_s=0.0)
    assert req.prompt == "hello world"
    assert req.expected_output_tokens == 64
    assert req.arrival_offset_s == 0.0
    assert req.metadata == {}


# ---------------------------------------------------------------------------
# Workload.min_context_tokens
# ---------------------------------------------------------------------------


def test_min_context_tokens_empty() -> None:
    wl = Workload(requests=[], duration_s=1.0)
    assert wl.min_context_tokens() == 512


def test_min_context_tokens_rounds_up_to_512() -> None:
    # 10-word prompt → 10 * 2.0 = 20 tokens input; 100 output → 120 total
    # 120 * 1.25 = 150 → round up to 512
    req = WorkloadRequest(
        prompt=" ".join(["word"] * 10), expected_output_tokens=100, arrival_offset_s=0.0
    )
    wl = Workload(requests=[req], duration_s=1.0)
    result = wl.min_context_tokens()
    assert result >= 512
    assert result % 512 == 0


def test_min_context_tokens_large_prompt() -> None:
    # 500-word prompt → 500 * 2.0 = 1000 input; 512 output → 1512 total
    # 1512 * 1.25 = 1890 → rounds up to 2048
    req = WorkloadRequest(
        prompt=" ".join([f"w{i}" for i in range(500)]),
        expected_output_tokens=512,
        arrival_offset_s=0.0,
    )
    wl = Workload(requests=[req], duration_s=1.0)
    result = wl.min_context_tokens()
    assert result >= 2048
    assert result % 512 == 0


# ---------------------------------------------------------------------------
# Workload.summary
# ---------------------------------------------------------------------------


def test_summary_empty() -> None:
    wl = Workload(requests=[], duration_s=10.0)
    s = wl.summary()
    assert s["request_count"] == 0
    assert s["requests_per_sec"] == 0.0
    assert s["prefix_cache_hit_estimate"] == 0.0


def test_summary_single_request() -> None:
    req = WorkloadRequest(
        prompt="hello world test", expected_output_tokens=128, arrival_offset_s=0.0
    )
    wl = Workload(requests=[req], duration_s=10.0)
    s = wl.summary()
    assert s["request_count"] == 1
    assert s["output_token_p50"] == 128
    assert s["output_token_p99"] == 128
    assert s["requests_per_sec"] == pytest.approx(0.1)


def test_summary_prefix_cache_estimate() -> None:
    # All requests share the same prefix → hit rate should be 1.0
    prefix = "system prompt " * 20
    requests = [
        WorkloadRequest(
            prompt=f"{prefix} query{i}", expected_output_tokens=64, arrival_offset_s=float(i)
        )
        for i in range(50)
    ]
    wl = Workload(requests=requests, duration_s=60.0)
    s = wl.summary()
    assert s["prefix_cache_hit_estimate"] > 0.5


# ---------------------------------------------------------------------------
# workload_from_jsonl
# ---------------------------------------------------------------------------


def test_workload_from_jsonl_basic(tmp_path: Path) -> None:
    lines = [
        {"prompt": "What is 2+2?", "expected_output_tokens": 10, "arrival_offset_s": 0.0},
        {
            "prompt": "What is the capital of France?",
            "expected_output_tokens": 20,
            "arrival_offset_s": 1.0,
        },
    ]
    jsonl = tmp_path / "workload.jsonl"
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))

    wl = workload_from_jsonl(jsonl, concurrency=4)
    assert len(wl.requests) == 2
    assert wl.requests[0].prompt == "What is 2+2?"
    assert wl.requests[1].expected_output_tokens == 20
    assert wl.concurrency == 4
    assert wl.duration_s == pytest.approx(2.0)


def test_workload_from_jsonl_stores_path(tmp_path: Path) -> None:
    jsonl = tmp_path / "wl.jsonl"
    jsonl.write_text(json.dumps({"prompt": "hi", "expected_output_tokens": 5}) + "\n")
    wl = workload_from_jsonl(jsonl)
    assert wl.metadata["path"] == str(jsonl.resolve())


def test_workload_from_jsonl_default_concurrency(tmp_path: Path) -> None:
    jsonl = tmp_path / "wl.jsonl"
    jsonl.write_text(json.dumps({"prompt": "hi", "expected_output_tokens": 5}) + "\n")
    wl = workload_from_jsonl(jsonl)
    assert wl.concurrency == 8


def test_workload_from_jsonl_optional_fields(tmp_path: Path) -> None:
    # arrival_offset_s and expected_output_tokens are optional (have defaults)
    jsonl = tmp_path / "wl.jsonl"
    jsonl.write_text(json.dumps({"prompt": "hello"}) + "\n")
    wl = workload_from_jsonl(jsonl)
    assert wl.requests[0].expected_output_tokens == 128
    assert wl.requests[0].arrival_offset_s == 0.0


def test_workload_from_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    jsonl = tmp_path / "wl.jsonl"
    jsonl.write_text(
        json.dumps({"prompt": "a", "expected_output_tokens": 1})
        + "\n\n"
        + json.dumps({"prompt": "b", "expected_output_tokens": 2})
        + "\n"
    )
    wl = workload_from_jsonl(jsonl)
    assert len(wl.requests) == 2


# ---------------------------------------------------------------------------
# Synthetic workload generators
# ---------------------------------------------------------------------------


def test_workload_synthetic_deterministic() -> None:
    wl1 = workload_synthetic(n_requests=20, seed=42)
    wl2 = workload_synthetic(n_requests=20, seed=42)
    assert [r.prompt for r in wl1.requests] == [r.prompt for r in wl2.requests]


def test_workload_synthetic_different_seeds() -> None:
    wl1 = workload_synthetic(n_requests=20, seed=0)
    wl2 = workload_synthetic(n_requests=20, seed=1)
    assert [r.prompt for r in wl1.requests] != [r.prompt for r in wl2.requests]


def test_workload_synthetic_request_count() -> None:
    wl = workload_synthetic(n_requests=50)
    assert len(wl.requests) == 50


def test_workload_synthetic_arrival_monotone() -> None:
    wl = workload_synthetic(n_requests=100)
    offsets = [r.arrival_offset_s for r in wl.requests]
    assert offsets == sorted(offsets)


def test_workload_shared_prefix_hit_rate() -> None:
    wl = workload_shared_prefix(n_requests=100, concurrency=8)
    s = wl.summary()
    # All requests share a long common prefix → should detect a high hit rate
    assert s["prefix_cache_hit_estimate"] > 0.5


def test_workload_shared_prefix_concurrency() -> None:
    wl = workload_shared_prefix(n_requests=50, concurrency=16)
    assert wl.concurrency == 16


def test_workload_concurrent_synthetic_concurrency() -> None:
    wl = workload_concurrent_synthetic(n_requests=30, concurrency=32)
    assert wl.concurrency == 32
    assert len(wl.requests) == 30


# ---------------------------------------------------------------------------
# Sample workload file
# ---------------------------------------------------------------------------


def test_sample_workload_jsonl_parses() -> None:
    """The bundled example workload should parse without errors."""
    sample = Path(__file__).parent.parent / "examples" / "sample_workload.jsonl"
    if not sample.exists():
        pytest.skip("examples/sample_workload.jsonl not found")
    wl = workload_from_jsonl(sample)
    assert len(wl.requests) >= 10
    for req in wl.requests:
        assert req.prompt
        assert req.expected_output_tokens > 0

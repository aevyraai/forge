# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for aevyra_forge.result — ForgeStore and ForgeRun."""

from __future__ import annotations

from pathlib import Path

from aevyra_forge.bench import BenchResult
from aevyra_forge.orchestrator import Experiment
from aevyra_forge.recipe import HardwareSpec, Recipe, VLLMConfig
from aevyra_forge.result import ForgeRun, ForgeStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_recipe(gen: int = 0, enable_prefix_caching: bool = False) -> Recipe:
    hw = HardwareSpec(vendor="nvidia", gpu_type="T4", count=1, memory_gb_per_gpu=16)
    cfg = VLLMConfig(enable_prefix_caching=enable_prefix_caching)
    return Recipe(
        model="meta-llama/Llama-3.2-1B-Instruct",
        hardware=hw,
        config=cfg,
        generation=gen,
    )


def _make_bench(throughput: float = 1000.0, accuracy: float = 0.99) -> BenchResult:
    return BenchResult(
        throughput_tokens_per_sec=throughput,
        p50_latency_ms=50.0,
        p99_latency_ms=150.0,
        ttft_ms=20.0,
        tpot_ms=5.0,
        peak_vram_mb=8192,
        max_concurrent_seqs=8,
        accuracy_score=accuracy,
        accuracy_delta_vs_baseline=None,
        recipe_id="test-recipe",
        workload_id="test-workload",
        bench_duration_s=30.0,
        status="PASS",
        error=None,
    )


def _make_experiment(
    exp_id: str = "exp001",
    score: float = 1.0,
    kept: bool = True,
    gen: int = 0,
    throughput: float = 1000.0,
) -> Experiment:
    return Experiment(
        id=exp_id,
        recipe=_make_recipe(gen=gen),
        bench_result=_make_bench(throughput=throughput),
        score=score,
        kept=kept,
        duration_s=30.0,
        started_at=0.0,
        ended_at=30.0,
        llm_tokens=500,
    )


# ---------------------------------------------------------------------------
# ForgeStore — new_run, get_run, find_incomplete_run
# ---------------------------------------------------------------------------


def test_forge_store_creates_run_dir(tmp_path: Path) -> None:
    store = ForgeStore(tmp_path / ".forge")
    run = store.new_run()
    assert run.path.exists()
    assert run.run_id == "001"


def test_forge_store_sequential_run_ids(tmp_path: Path) -> None:
    store = ForgeStore(tmp_path / ".forge")
    r1 = store.new_run()
    r1.path.mkdir(parents=True, exist_ok=True)
    r2 = store.new_run()
    assert r1.run_id == "001"
    assert r2.run_id == "002"


def test_forge_store_latest_run_none_when_empty(tmp_path: Path) -> None:
    store = ForgeStore(tmp_path / ".forge")
    assert store.latest_run() is None


def test_forge_store_latest_run(tmp_path: Path) -> None:
    store = ForgeStore(tmp_path / ".forge")
    store.new_run()
    r2 = store.new_run()
    latest = store.latest_run()
    assert latest is not None
    assert latest.run_id == r2.run_id


def test_forge_store_find_incomplete_none(tmp_path: Path) -> None:
    store = ForgeStore(tmp_path / ".forge")
    assert store.find_incomplete_run() is None


def test_forge_store_find_incomplete_run(tmp_path: Path) -> None:
    store = ForgeStore(tmp_path / ".forge")
    run = store.new_run()
    # Write experiments.jsonl but NOT completed.json → interrupted
    run.append(_make_experiment())
    incomplete = store.find_incomplete_run()
    assert incomplete is not None
    assert incomplete.run_id == run.run_id


def test_forge_store_find_incomplete_ignores_complete(tmp_path: Path) -> None:
    store = ForgeStore(tmp_path / ".forge")
    run = store.new_run()
    run.append(_make_experiment())
    run.save_completion(best_score=1.0, n_kept=1, n_total=1, wall_time_s=60.0)
    assert store.find_incomplete_run() is None


# ---------------------------------------------------------------------------
# ForgeRun.save_config / .config
# ---------------------------------------------------------------------------


def test_forge_run_save_and_read_config(tmp_path: Path) -> None:
    run = ForgeRun(tmp_path / "run1")
    run.path.mkdir()
    run.save_config(
        model="mymodel",
        hardware_label="nvidia/T4x1",
        workload_id="wl1",
        forge_config_dict={"max_experiments": 10},
        llm_provider="anthropic/claude-sonnet-4-6",
        device="cuda",
        workload_path="/data/wl.jsonl",
        concurrency=8,
    )
    cfg = run.config()
    assert cfg is not None
    assert cfg["model"] == "mymodel"
    assert cfg["device"] == "cuda"
    assert cfg["workload_path"] == "/data/wl.jsonl"
    assert cfg["concurrency"] == 8
    assert cfg["llm_provider"] == "anthropic/claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# ForgeRun.append / .history / .best
# ---------------------------------------------------------------------------


def test_forge_run_append_and_history(tmp_path: Path) -> None:
    run = ForgeRun(tmp_path / "run1")
    run.path.mkdir()

    e1 = _make_experiment("e1", score=1.0, kept=True)
    e2 = _make_experiment("e2", score=1.5, kept=True)
    e3 = _make_experiment("e3", score=0.8, kept=False)

    run.append(e1)
    run.append(e2)
    run.append(e3)

    history = run.history()
    assert len(history) == 3
    assert history[0].id == "e1"
    assert history[1].id == "e2"
    assert history[2].id == "e3"


def test_forge_run_best_selects_highest_kept(tmp_path: Path) -> None:
    run = ForgeRun(tmp_path / "run1")
    run.path.mkdir()

    run.append(_make_experiment("e1", score=1.0, kept=True))
    run.append(_make_experiment("e2", score=2.0, kept=True))
    run.append(_make_experiment("e3", score=3.0, kept=False))  # not kept

    best = run.best()
    assert best is not None
    assert best.id == "e2"
    assert best.score == 2.0


def test_forge_run_best_none_when_no_kept(tmp_path: Path) -> None:
    run = ForgeRun(tmp_path / "run1")
    run.path.mkdir()
    run.append(_make_experiment("e1", score=1.0, kept=False))
    assert run.best() is None


def test_forge_run_history_empty_before_append(tmp_path: Path) -> None:
    run = ForgeRun(tmp_path / "run1")
    run.path.mkdir()
    assert run.history() == []


# ---------------------------------------------------------------------------
# ForgeRun status helpers
# ---------------------------------------------------------------------------


def test_forge_run_status_running(tmp_path: Path) -> None:
    run = ForgeRun(tmp_path / "run1")
    run.path.mkdir()
    assert run.status() == "running"


def test_forge_run_status_interrupted(tmp_path: Path) -> None:
    run = ForgeRun(tmp_path / "run1")
    run.path.mkdir()
    run.append(_make_experiment())
    assert run.status() == "interrupted"
    assert run.is_interrupted() is True
    assert run.is_complete() is False


def test_forge_run_status_completed(tmp_path: Path) -> None:
    run = ForgeRun(tmp_path / "run1")
    run.path.mkdir()
    run.append(_make_experiment())
    run.save_completion(best_score=1.0, n_kept=1, n_total=1, wall_time_s=120.0)
    assert run.status() == "completed"
    assert run.is_complete() is True
    assert run.is_interrupted() is False


# ---------------------------------------------------------------------------
# ForgeRun.render_tsv
# ---------------------------------------------------------------------------


def test_forge_run_render_tsv_header(tmp_path: Path) -> None:
    run = ForgeRun(tmp_path / "run1")
    run.path.mkdir()
    tsv = run.render_tsv()
    header = tsv.splitlines()[0]
    assert "exp" in header
    assert "score" in header
    assert "throughput" in header
    assert "kept" in header


def test_forge_run_render_tsv_with_data(tmp_path: Path) -> None:
    run = ForgeRun(tmp_path / "run1")
    run.path.mkdir()
    run.append(_make_experiment("e1", score=1.23, kept=True))
    tsv = run.render_tsv()
    lines = tsv.splitlines()
    assert len(lines) == 2  # header + 1 row
    assert "1.2300" in lines[1]
    assert "✓" in lines[1]


def test_forge_run_render_tsv_not_kept(tmp_path: Path) -> None:
    run = ForgeRun(tmp_path / "run1")
    run.path.mkdir()
    run.append(_make_experiment("e1", score=0.5, kept=False))
    tsv = run.render_tsv()
    assert "✗" in tsv.splitlines()[1]


# ---------------------------------------------------------------------------
# ForgeStore.list_runs
# ---------------------------------------------------------------------------


def test_forge_store_list_runs(tmp_path: Path) -> None:
    store = ForgeStore(tmp_path / ".forge")
    run = store.new_run()
    run.append(_make_experiment("e1", score=1.0, kept=True))
    run.save_completion(best_score=1.0, n_kept=1, n_total=1, wall_time_s=60.0)

    rows = store.list_runs()
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["n_experiments"] == 1

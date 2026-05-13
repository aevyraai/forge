# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Run persistence — checkpointing, resume, and multi-run versioning.

Manages a ``.forge/`` directory that stores every optimization run with its
config, experiments, and results.  Each run gets a sequential ID and a
timestamped directory::

    .forge/
      runs/
        001_2026-05-13T04-10-00/
          config.json        — model, hardware, workload metadata
          experiments.jsonl  — one line per experiment (append-only)
          experiments.json   — structured table array (re-written each append)
          experiments.tsv    — human-readable summary (re-written each append)
          best_recipe.yaml   — best recipe so far (updated when score improves)
          completed.json     — written on clean finish; absent = interrupted
        002_2026-05-13T14-05-00/
          ...

A run that has ``experiments.jsonl`` but no ``completed.json`` was interrupted
and can be resumed.  :class:`ForgeStore` finds the latest such run via
:meth:`~ForgeStore.find_incomplete_run`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aevyra_forge.orchestrator import Experiment


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")


def _write_json(path: Path, data: Any) -> None:
    """Atomic JSON write — write to .tmp then rename."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.rename(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# ForgeRun — handle for one run's on-disk directory
# ---------------------------------------------------------------------------


class ForgeRun:
    """Handle for one Forge optimization run's directory.

    Created by :class:`ForgeStore` — don't instantiate directly.

    The run directory is the unit of resume: everything needed to continue
    a crashed run lives here.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.run_id: str = path.name.split("_")[0]
        self._jsonl_path = path / "experiments.jsonl"
        self._json_path = path / "experiments.json"
        self._tsv_path = path / "experiments.tsv"
        self._best_path = path / "best_recipe.yaml"
        self._config_path = path / "config.json"
        self._completed_path = path / "completed.json"

    # ------------------------------------------------------------------
    # Config (written once at run start)
    # ------------------------------------------------------------------

    def save_config(
        self,
        *,
        model: str,
        hardware_label: str,
        workload_id: str,
        forge_config_dict: dict[str, Any],
        llm_provider: str = "",
        device: str = "cuda",
        workload_path: str = "",
        concurrency: int = 8,
    ) -> None:
        """Write run metadata. Called once when a new run starts."""
        self.path.mkdir(parents=True, exist_ok=True)
        _write_json(
            self._config_path,
            {
                "run_id": self.run_id,
                "model": model,
                "hardware": hardware_label,
                "device": device,
                "workload_id": workload_id,
                "workload_path": workload_path,
                "concurrency": concurrency,
                "llm_provider": llm_provider,
                "forge_config": forge_config_dict,
                "started_at": _now_iso(),
            },
        )

    def config(self) -> dict[str, Any] | None:
        if not self._config_path.exists():
            return None
        return _read_json(self._config_path)

    # ------------------------------------------------------------------
    # Experiment append (called after every experiment)
    # ------------------------------------------------------------------

    def append(self, exp: Experiment) -> None:
        """Atomic JSONL append + re-render JSON table, TSV, best_recipe."""
        import dataclasses

        self.path.mkdir(parents=True, exist_ok=True)
        record = dataclasses.asdict(exp)
        with self._jsonl_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        self._tsv_path.write_text(self.render_tsv())
        _write_json(self._json_path, self._render_json_table())
        best = self.best()
        if best is not None:
            self._best_path.write_text(best.recipe.to_yaml())

    # ------------------------------------------------------------------
    # Completion marker
    # ------------------------------------------------------------------

    def save_completion(
        self,
        *,
        best_score: float,
        n_kept: int,
        n_total: int,
        wall_time_s: float,
    ) -> None:
        """Write completed.json.  Marks the run as cleanly finished."""
        _write_json(
            self._completed_path,
            {
                "run_id": self.run_id,
                "best_score": best_score,
                "n_kept": n_kept,
                "n_total": n_total,
                "wall_time_s": round(wall_time_s, 1),
                "completed_at": _now_iso(),
            },
        )

    def completion(self) -> dict[str, Any] | None:
        if not self._completed_path.exists():
            return None
        return _read_json(self._completed_path)

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def is_complete(self) -> bool:
        return self._completed_path.exists()

    def is_interrupted(self) -> bool:
        return self._jsonl_path.exists() and not self._completed_path.exists()

    def status(self) -> str:
        if self.is_complete():
            return "completed"
        if self._jsonl_path.exists():
            return "interrupted"
        return "running"

    def summary_row(self) -> dict[str, Any]:
        """One-row summary dict for :meth:`ForgeStore.list_runs`."""
        cfg = self.config() or {}
        done = self.completion() or {}
        history = self.history()
        n_kept = sum(1 for e in history if e.kept)
        best_score = max((e.score or 0.0 for e in history if e.kept), default=0.0)
        return {
            "run_id": self.run_id,
            "status": self.status(),
            "model": cfg.get("model", "?"),
            "hardware": cfg.get("hardware", "?"),
            "n_experiments": len(history),
            "n_kept": n_kept,
            "best_score": done.get("best_score", best_score),
            "wall_time_s": done.get("wall_time_s"),
            "started_at": cfg.get("started_at", ""),
            "completed_at": done.get("completed_at", ""),
        }

    # ------------------------------------------------------------------
    # Read back
    # ------------------------------------------------------------------

    def history(self) -> list[Experiment]:
        """Read all experiments back from disk."""
        from aevyra_forge.agent import AgentDecision
        from aevyra_forge.bench import BenchResult
        from aevyra_forge.orchestrator import Experiment as Exp
        from aevyra_forge.recipe import Recipe

        if not self._jsonl_path.exists():
            return []
        exps: list[Experiment] = []
        for line in self._jsonl_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            bench = BenchResult(**d["bench_result"]) if d.get("bench_result") else None
            decision = None
            if d.get("agent_decision"):
                ad = d["agent_decision"]
                decision = AgentDecision(
                    rationale=ad["rationale"],
                    mutation=ad["mutation"],
                    expected_throughput_delta_pct=ad.get("expected_throughput_delta_pct"),
                    expected_accuracy_delta=ad.get("expected_accuracy_delta"),
                    raw_response=ad.get("raw_response", ""),
                )
            exps.append(
                Exp(
                    id=d["id"],
                    recipe=Recipe.from_dict(d["recipe"]),
                    bench_result=bench,
                    agent_decision=decision,
                    score=d.get("score"),
                    kept=d.get("kept", False),
                    duration_s=d.get("duration_s", 0.0),
                    started_at=d.get("started_at", 0.0),
                    ended_at=d.get("ended_at", 0.0),
                    llm_tokens=d.get("llm_tokens", 0),
                )
            )
        return exps

    def best(self) -> Experiment | None:
        """Return the highest-scoring kept experiment."""
        kept = [e for e in self.history() if e.kept and e.score is not None]
        return max(kept, key=lambda e: e.score or 0.0) if kept else None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_tsv(self) -> str:
        """Human-readable summary table as TSV."""
        header = "\t".join(
            ["exp", "id", "gen", "score", "throughput", "p99_ms", "accuracy", "kept", "rationale"]
        )
        rows = [header]
        for i, exp in enumerate(self.history()):
            br = exp.bench_result
            rationale = (
                exp.agent_decision.rationale[:60].replace("\t", " ") if exp.agent_decision else ""
            )
            rows.append(
                "\t".join(
                    [
                        str(i),
                        exp.id,
                        str(exp.recipe.generation),
                        f"{exp.score:.4f}" if exp.score is not None else "-",
                        f"{br.throughput_tokens_per_sec:.1f}" if br else "-",
                        f"{br.p99_latency_ms:.0f}" if br else "-",
                        (
                            f"{br.accuracy_score:.3f}"
                            if br and br.accuracy_score is not None
                            else "-"
                        ),
                        "✓" if exp.kept else "✗",
                        rationale,
                    ]
                )
            )
        return "\n".join(rows)

    def _render_json_table(self) -> list[dict[str, Any]]:
        """Structured table as a list of dicts — machine-readable experiments.json."""
        rows = []
        for i, exp in enumerate(self.history()):
            br = exp.bench_result
            changes: dict[str, Any] = {}
            if exp.agent_decision and exp.agent_decision.mutation.get("changes"):
                changes = exp.agent_decision.mutation["changes"]
            rows.append(
                {
                    "n": i,
                    "id": exp.id,
                    "generation": exp.recipe.generation,
                    "score": exp.score,
                    "throughput_tokens_per_sec": br.throughput_tokens_per_sec if br else None,
                    "p50_latency_ms": br.p50_latency_ms if br else None,
                    "p99_latency_ms": br.p99_latency_ms if br else None,
                    "ttft_ms": br.ttft_ms if br else None,
                    "status": br.status if br else None,
                    "kept": exp.kept,
                    "duration_s": exp.duration_s,
                    "llm_tokens": exp.llm_tokens,
                    "changes": changes,
                    "rationale": exp.agent_decision.rationale if exp.agent_decision else None,
                    "error": br.error if br else None,
                }
            )
        return rows

    def render_pareto(self) -> str:
        return ""


# Backward-compat alias — code that imports ExperimentStore still works
ExperimentStore = ForgeRun


# ---------------------------------------------------------------------------
# ForgeStore — manages a directory of runs
# ---------------------------------------------------------------------------


@dataclass
class RunSummary:
    """Lightweight summary for listing / comparison."""

    run_id: str
    status: str
    model: str
    hardware: str
    n_experiments: int
    n_kept: int
    best_score: float
    wall_time_s: float | None
    started_at: str
    completed_at: str


class ForgeStore:
    """Manages a directory of Forge optimization runs.

    Args:
        root: Root directory.  Defaults to ``.forge`` in the current
              working directory.  Created on first use.

    Layout::

        .forge/
          runs/
            001_2026-05-13T04-10-00/
            002_2026-05-13T14-05-00/

    Usage::

        store = ForgeStore()                  # uses .forge/
        orch  = Orchestrator(..., store=store)
        best, history = orch.run()            # creates 001_.../ automatically

        # resume latest interrupted run
        best, history = orch.resume()

        # audit
        for row in store.list_runs():
            print(row)
    """

    def __init__(self, root: str | Path = ".forge") -> None:
        self.root = Path(root)
        self.runs_dir = self.root / "runs"

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def new_run(self) -> ForgeRun:
        """Create a new numbered + timestamped run directory."""
        run_id = self._next_run_id()
        timestamp = _now_iso()
        path = self.runs_dir / f"{run_id}_{timestamp}"
        path.mkdir(parents=True, exist_ok=True)
        return ForgeRun(path)

    def get_run(self, run_id: str) -> ForgeRun | None:
        """Look up a run by ID (e.g. ``'001'``). Returns ``None`` if not found."""
        for d in self._run_dirs():
            if d.name.startswith(f"{run_id}_") or d.name == run_id:
                return ForgeRun(d)
        return None

    def latest_run(self) -> ForgeRun | None:
        """Return the most recently created run (complete or not)."""
        dirs = self._run_dirs()
        return ForgeRun(dirs[-1]) if dirs else None

    def find_incomplete_run(self) -> ForgeRun | None:
        """Return the most recent interrupted run (has experiments, no completed.json).

        Used to implement ``--resume``: call this, then pass the returned
        ``ForgeRun`` to :meth:`Orchestrator.resume`.
        """
        for d in reversed(self._run_dirs()):
            run = ForgeRun(d)
            if run.is_interrupted():
                return run
        return None

    def list_runs(self) -> list[dict[str, Any]]:
        """Return a summary row for every run, newest first."""
        return [ForgeRun(d).summary_row() for d in reversed(self._run_dirs())]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_dirs(self) -> list[Path]:
        if not self.runs_dir.exists():
            return []
        return sorted(
            [d for d in self.runs_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

    def _next_run_id(self) -> str:
        ids = []
        for d in self._run_dirs():
            m = re.match(r"^(\d+)_", d.name)
            if m:
                ids.append(int(m.group(1)))
        return f"{(max(ids, default=0) + 1):03d}"


__all__ = [
    "ExperimentStore",
    "ForgeRun",
    "ForgeStore",
    "RunSummary",
]

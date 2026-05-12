# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Experiment store — append-only JSONL log + human-readable TSV summary.

See AGENT.md → "Module-by-module spec → result.py".

This mirrors AutoKernel's results.tsv pattern: every experiment is
durably logged so a crashed run can resume, a finished run can be
diffed against another, and the audit trail is git-friendly text.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from aevyra_forge.orchestrator import Experiment


logger = logging.getLogger(__name__)


class ExperimentStore:
    """Append-only experiment log under a single run directory.

    Layout::

        runs/2026-05-11T08-32-00/
          config.json         — model, hardware, workload, playbook digest
          experiments.jsonl   — one line per experiment (recipe diff + bench result + agent decision)
          experiments.tsv     — rendered human-readable summary
          best_recipe.yaml    — pointer to the best recipe so far
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self._jsonl_path = run_dir / "experiments.jsonl"
        self._tsv_path = run_dir / "experiments.tsv"
        self._best_path = run_dir / "best_recipe.yaml"

    def append(self, exp: Experiment) -> None:
        """Atomic JSONL append. Re-renders TSV and updates best_recipe."""
        import dataclasses
        import json as _json

        self.run_dir.mkdir(parents=True, exist_ok=True)
        record = dataclasses.asdict(exp)
        with self._jsonl_path.open("a") as f:
            f.write(_json.dumps(record) + "\n")
        self._tsv_path.write_text(self.render_tsv())
        best = self.best()
        if best is not None:
            self._best_path.write_text(best.recipe.to_yaml())

    def history(self) -> list[Experiment]:
        """Read all experiments back from disk."""
        import json as _json
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
            d = _json.loads(line)
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
                )
            )
        return exps

    def best(self) -> Experiment | None:
        """Return the highest-scoring kept experiment."""
        kept = [e for e in self.history() if e.kept and e.score is not None]
        return max(kept, key=lambda e: e.score or 0.0) if kept else None

    def render_tsv(self) -> str:
        """Human-readable summary table."""
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
                        f"{br.accuracy_score:.3f}" if br and br.accuracy_score is not None else "-",
                        "✓" if exp.kept else "✗",
                        rationale,
                    ]
                )
            )
        return "\n".join(rows)

    def render_pareto(self) -> str:
        return ""

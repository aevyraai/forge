# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""The main loop — Amdahl-style scheduler over the keep/revert experiments.

See AGENT.md → "Key concepts → The orchestrator" and "v0 scope".

The orchestrator does NOT pick the next mutation — the agent does.
The orchestrator owns:

- Scheduling (which layer to spend the next experiment on)
- Layer escalation (when to jump L1 → L2 → L3)
- Budget enforcement (max experiments, wall-clock, dollars)
- Convergence detection
- Statistical significance gates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from aevyra_forge.bench import BenchResult
from aevyra_forge.recipe import HardwareSpec, Recipe
from aevyra_forge.workload import Workload


if TYPE_CHECKING:
    from aevyra_forge.agent import AgentDecision
    from aevyra_forge.llm import LLMFn
    from aevyra_forge.playbook import Playbook
    from aevyra_forge.result import ExperimentStore


logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    id: str
    recipe: Recipe
    bench_result: BenchResult | None = None
    agent_decision: "AgentDecision | None" = None
    score: float | None = None
    kept: bool = False
    duration_s: float = 0.0
    started_at: float = 0.0
    ended_at: float = 0.0


@dataclass
class ForgeConfig:
    """Budget + termination knobs."""

    max_experiments: int = 50
    max_wall_clock_hours: float = 12.0
    max_dollars: float | None = None
    accuracy_floor: float = 0.99
    min_improvement_pct: float = 1.0
    convergence_window: int = 5
    layer_escalation: bool = True
    dry_run: bool = False
    work_dir: Path = field(default_factory=lambda: Path("./runs/current"))


class Orchestrator:
    """The autotune loop. Construct, call ``.run()``, get back the best recipe + history."""

    def __init__(
        self,
        *,
        model: str,
        hardware: HardwareSpec,
        workload: Workload,
        playbook: "Playbook",
        llm: "LLMFn",
        store: "ExperimentStore",
        forge_config: ForgeConfig | None = None,
    ) -> None:
        self.model = model
        self.hardware = hardware
        self.workload = workload
        self.playbook = playbook
        self.llm = llm
        self.store = store
        self.cfg = forge_config or ForgeConfig()

    def run(self) -> tuple[Recipe, list[Experiment]]:
        """Main loop.

        1. Build the baseline recipe (config.baseline_config).
        2. Run baseline through runner + bench. Log experiment 0.
        3. While not converged and budget not exhausted:
           a. Build context: history, current best, layer state.
           b. agent.propose_next_experiment(...) → mutation
           c. Apply mutation via config.mutate (or quant.mutate, kernel.mutate)
           d. Boot runner, bench, compute score
           e. Keep or revert (compare against current best)
           f. Log experiment
           g. Update layer state (escalate if Layer 1 saturated and escalation enabled)
        4. Return (best_recipe, history)

        Failure handling: bench CRASH/TIMEOUT becomes a kept=False
        experiment with score=0. The agent sees it in history and
        avoids the region.
        """
        import time

        from aevyra_forge import config as config_mod
        from aevyra_forge.bench import score as bench_score
        from aevyra_forge.recipe import Recipe

        self._run_start = time.time()
        current_layer = "config"

        # Build baseline recipe
        baseline_cfg = config_mod.baseline_config(self.hardware)
        baseline_recipe = Recipe(
            model=self.model,
            hardware=self.hardware,
            config=baseline_cfg,
        )

        # Run baseline experiment
        logger.info("forge ┌─ experiment 0/%d  [baseline]", self.cfg.max_experiments)
        baseline_exp = self._run_one(baseline_recipe, agent_decision=None)
        baseline_exp.kept = True
        self.store.append(baseline_exp)

        history = self.store.history()
        current_recipe = baseline_recipe
        best_score = baseline_exp.score or 0.0

        logger.info(
            "forge └─ baseline  score=%.1f tok/s  p99=%.0fms  status=%s",
            best_score,
            baseline_exp.bench_result.p99_latency_ms if baseline_exp.bench_result else 0.0,
            baseline_exp.bench_result.status if baseline_exp.bench_result else "?",
        )

        while not self._is_converged(history) and not self._budget_exhausted(history):
            # Maybe escalate layer
            if self.cfg.layer_escalation:
                current_layer = self._should_escalate(history, current_layer)

            # Ask agent for next mutation
            exp_n = len(history)
            logger.info(
                "forge │  experiment %d/%d — asking agent...",
                exp_n, self.cfg.max_experiments,
            )
            try:
                from aevyra_forge import agent as agent_mod
                decision = agent_mod.propose_next_experiment(
                    history=history,
                    playbook=self.playbook,
                    current_recipe=current_recipe,
                    workload=self.workload,
                    llm=self.llm,
                )
            except Exception as exc:
                logger.warning("forge │  agent call failed: %s — skipping", exc)
                continue

            # Empty changes = agent says converged
            if not decision.mutation.get("changes"):
                logger.info("forge │  agent returned empty mutation — converged.")
                break

            logger.info(
                "forge ┌─ experiment %d/%d",
                exp_n, self.cfg.max_experiments,
            )
            logger.info("forge │  rationale : %s", decision.rationale)
            logger.info("forge │  mutation  : %s", decision.mutation.get("changes", {}))

            # Apply mutation
            try:
                from aevyra_forge import config as config_mod
                candidate = config_mod.mutate(current_recipe, decision.mutation)
            except (ValueError, NotImplementedError) as exc:
                logger.warning("Mutation rejected: %s", exc)
                # Log a failed experiment so the agent sees this region is invalid
                exp = Experiment(
                    id=f"rej-{len(history)}",
                    recipe=current_recipe,
                    agent_decision=decision,
                    score=0.0,
                    kept=False,
                )
                self.store.append(exp)
                history = self.store.history()
                continue

            # Run experiment
            exp = self._run_one(candidate, agent_decision=decision)

            # Keep or revert
            improvement_threshold = best_score * (1 + self.cfg.min_improvement_pct / 100)
            p99 = exp.bench_result.p99_latency_ms if exp.bench_result else 0.0
            dur = exp.duration_s
            if exp.score is not None and exp.score > improvement_threshold:
                exp.kept = True
                gain = 100 * (exp.score - best_score) / max(best_score, 1e-9)
                current_recipe = candidate
                best_score = exp.score
                logger.info(
                    "forge └─ ✓ KEPT      score=%.1f tok/s  p99=%.0fms  +%.1f%%  duration=%.0fs",
                    exp.score, p99, gain, dur,
                )
            else:
                exp.kept = False
                logger.info(
                    "forge └─ ✗ REVERTED  score=%.1f tok/s  p99=%.0fms  best=%.1f  duration=%.0fs",
                    exp.score or 0.0, p99, best_score, dur,
                )

            self.store.append(exp)
            history = self.store.history()

        best_exp = self.store.best()
        best_recipe = best_exp.recipe if best_exp else baseline_recipe
        baseline_score = history[0].score or 0.0
        total_gain = 100 * (best_score - baseline_score) / max(baseline_score, 1e-9)
        elapsed = time.time() - self._run_start
        kept_count = sum(1 for e in history if e.kept)
        logger.info("forge ══════════════════════════════════════════")
        logger.info("forge   total experiments : %d", len(history))
        logger.info("forge   kept              : %d", kept_count)
        logger.info("forge   baseline score    : %.1f tok/s", baseline_score)
        logger.info("forge   best score        : %.1f tok/s  (+%.1f%%)", best_score, total_gain)
        logger.info("forge   wall time         : %.0f min", elapsed / 60)
        logger.info("forge   best recipe gen   : %d  id=%s",
                    best_recipe.generation, best_recipe.id)
        logger.info("forge ══════════════════════════════════════════")

        analysis = self._generate_analysis(history, baseline_score, best_score, elapsed)
        if analysis:
            print("\n" + "─" * 60)
            print("  Forge Analysis")
            print("─" * 60)
            print(analysis)
            print("─" * 60 + "\n")

        return best_recipe, history

    def resume(self) -> tuple[Recipe, list[Experiment]]:
        """Continue from an existing run directory. Re-loads store.history()."""
        import time

        history = self.store.history()
        if not history:
            logger.warning("No existing experiments found in store — starting fresh.")
            return self.run()

        best_exp = self.store.best()
        if best_exp is None:
            logger.warning("No kept experiments found — starting fresh.")
            return self.run()

        self._run_start = time.time()
        current_recipe = best_exp.recipe
        best_score = best_exp.score or 0.0
        current_layer = "config"

        logger.info(
            "Resuming from experiment %s with best_score=%.4f (%d experiments in history)",
            best_exp.id, best_score, len(history),
        )

        while not self._is_converged(history) and not self._budget_exhausted(history):
            if self.cfg.layer_escalation:
                current_layer = self._should_escalate(history, current_layer)

            try:
                from aevyra_forge import agent as agent_mod
                decision = agent_mod.propose_next_experiment(
                    history=history,
                    playbook=self.playbook,
                    current_recipe=current_recipe,
                    workload=self.workload,
                    llm=self.llm,
                )
            except Exception as exc:
                logger.warning("Agent call failed: %s — skipping", exc)
                continue

            if not decision.mutation.get("changes"):
                logger.info("Agent returned empty mutation — converged.")
                break

            try:
                from aevyra_forge import config as config_mod
                candidate = config_mod.mutate(current_recipe, decision.mutation)
            except (ValueError, NotImplementedError) as exc:
                logger.warning("Mutation rejected: %s", exc)
                continue

            exp = self._run_one(candidate, agent_decision=decision)

            improvement_threshold = best_score * (1 + self.cfg.min_improvement_pct / 100)
            if exp.score is not None and exp.score > improvement_threshold:
                exp.kept = True
                current_recipe = candidate
                best_score = exp.score
            else:
                exp.kept = False

            self.store.append(exp)
            history = self.store.history()

        best_exp = self.store.best()
        best_recipe = best_exp.recipe if best_exp else current_recipe
        return best_recipe, history

    def _run_one(
        self,
        recipe: Recipe,
        *,
        agent_decision: "AgentDecision | None",
    ) -> Experiment:
        """Boot server, benchmark, score. Never raises — failures captured in BenchResult."""
        import time

        from aevyra_forge.bench import benchmark, score as bench_score
        from aevyra_forge.runner import VLLMRunner

        t_start = time.time()
        exp = Experiment(
            id=recipe.id,
            recipe=recipe,
            agent_decision=agent_decision,
            started_at=t_start,
        )

        try:
            with VLLMRunner(
                recipe,
                self.cfg.work_dir,
                dry_run=self.cfg.dry_run,
            ) as runner:
                bench_result = benchmark(
                    server_url=runner.url(),
                    workload=self.workload,
                    recipe_id=recipe.id,
                    workload_id=self.workload.metadata.get("id", "default"),
                    accuracy_check=True,
                    timeout_s=600,
                    dry_run=self.cfg.dry_run,
                )
        except Exception as exc:
            logger.error("Runner/bench failed for recipe %s: %s", recipe.id, exc)
            bench_result = BenchResult(
                throughput_tokens_per_sec=0.0,
                p50_latency_ms=0.0,
                p99_latency_ms=0.0,
                ttft_ms=0.0,
                tpot_ms=0.0,
                peak_vram_mb=0,
                max_concurrent_seqs=0,
                accuracy_score=None,
                accuracy_delta_vs_baseline=None,
                recipe_id=recipe.id,
                workload_id=self.workload.metadata.get("id", "default"),
                bench_duration_s=time.time() - t_start,
                status="CRASH",
                error=str(exc),
            )

        t_end = time.time()
        exp.bench_result = bench_result
        exp.score = bench_score(bench_result, accuracy_floor=self.cfg.accuracy_floor)
        exp.duration_s = t_end - t_start
        exp.ended_at = t_end
        return exp

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _should_escalate(self, history: list[Experiment], current_layer: str) -> str:
        """Return the layer to use for the next experiment.

        Returns ``current_layer`` if no escalation is warranted, or
        the next layer up if Amdahl says so. Layer order:
        ``config`` → ``quant`` → ``kernel``.

        Escalation rule: if the last ``convergence_window`` config experiments
        are all kept=False, the config layer has saturated → move to quant.
        quant → kernel follows the same pattern. v0.2+ only; for now quant
        and kernel are stubs, so we stay on config.
        """
        _layer_order = ["config", "quant", "kernel"]
        if not self.cfg.layer_escalation:
            return current_layer

        idx = _layer_order.index(current_layer) if current_layer in _layer_order else 0
        if idx >= len(_layer_order) - 1:
            return current_layer  # already at kernel level

        # Count recent experiments at current layer
        recent = [
            e for e in history
            if e.agent_decision and e.agent_decision.mutation.get("layer") == current_layer
        ][-self.cfg.convergence_window:]

        if len(recent) >= self.cfg.convergence_window and not any(e.kept for e in recent):
            next_layer = _layer_order[idx + 1]
            logger.info(
                "Layer %s saturated (%d consecutive non-improvements) — escalating to %s",
                current_layer, self.cfg.convergence_window, next_layer,
            )
            return next_layer

        return current_layer

    def _is_converged(self, history: list[Experiment]) -> bool:
        """True iff the last ``convergence_window`` experiments haven't
        improved the score by ``min_improvement_pct``.
        """
        if len(history) < self.cfg.convergence_window:
            return False

        recent = history[-self.cfg.convergence_window:]
        # If any experiment in the window was kept, we haven't converged
        if any(e.kept for e in recent):
            return False

        # Also check if all scores are essentially the same
        scores = [e.score for e in recent if e.score is not None]
        if not scores:
            return False

        best_in_window = max(scores)
        baseline_best = max(
            (e.score for e in history if e.kept and e.score is not None),
            default=0.0,
        )
        if baseline_best <= 0:
            return False

        improvement = (best_in_window - baseline_best) / baseline_best * 100
        if improvement < self.cfg.min_improvement_pct:
            logger.info(
                "Convergence detected: best improvement in last %d experiments = %.2f%% < %.1f%%",
                self.cfg.convergence_window, improvement, self.cfg.min_improvement_pct,
            )
            return True

        return False

    def _budget_exhausted(self, history: list[Experiment]) -> bool:
        """Check experiments, wall-clock, and dollar budgets."""
        import time

        if len(history) >= self.cfg.max_experiments:
            logger.info(
                "Budget exhausted: %d experiments reached (max=%d)",
                len(history), self.cfg.max_experiments,
            )
            return True

        run_start = getattr(self, "_run_start", None)
        if run_start is not None:
            elapsed_hours = (time.time() - run_start) / 3600
            if elapsed_hours >= self.cfg.max_wall_clock_hours:
                logger.info(
                    "Budget exhausted: %.2f hours elapsed (max=%.1f)",
                    elapsed_hours, self.cfg.max_wall_clock_hours,
                )
                return True

        if self.cfg.max_dollars is not None:
            tokens_used = getattr(self.llm, "tokens_used", 0)
            # Rough estimate: $3/M input tokens (Sonnet pricing)
            estimated_cost = tokens_used / 1_000_000 * 3.0
            if estimated_cost >= self.cfg.max_dollars:
                logger.info(
                    "Budget exhausted: estimated cost $%.4f >= $%.2f limit",
                    estimated_cost, self.cfg.max_dollars,
                )
                return True

        return False

    def _generate_analysis(
        self,
        history: list[Experiment],
        baseline_score: float,
        best_score: float,
        elapsed_s: float,
    ) -> str:
        """Call the agent LLM to produce a plain-English post-run analysis.

        Returns the analysis string, or an empty string if the LLM call fails.
        """
        try:
            kept = [e for e in history if e.kept]
            reverted = [e for e in history if not e.kept and e.agent_decision is not None]

            # Summarise what mutations were tried
            tried: list[str] = []
            for e in history:
                if e.agent_decision:
                    changes = e.agent_decision.mutation.get("changes", {})
                    for k, v in changes.items():
                        tried.append(f"{k}={v} ({'kept' if e.kept else 'reverted'})")

            tried_str = "; ".join(tried[:20]) or "none"
            improvement_pct = 100 * (best_score - baseline_score) / max(baseline_score, 1e-9)
            hw = self.hardware

            prompt = f"""\
You are analyzing the results of an automated vLLM deployment optimization run.

Hardware: {hw.vendor} {hw.gpu_type} x{hw.count} ({hw.memory_gb_per_gpu} GB VRAM each)
Model: {self.model}
Total experiments: {len(history)}
Kept (improved): {len(kept)}
Reverted: {len(reverted)}
Baseline throughput: {baseline_score:.1f} tok/s
Best throughput: {best_score:.1f} tok/s
Improvement: {improvement_pct:.1f}%
Wall time: {elapsed_s / 60:.0f} min
Mutations tried: {tried_str}

Write a short analysis (4-6 sentences) for the user. Cover:
1. What the results mean (e.g. baseline=best means the hardware is already near its ceiling for this config, or significant gain means chunked prefill / prefix caching helped)
2. Why configs did or didn't move the needle (memory bandwidth ceiling, KV-cache pressure, batch size already optimal, etc.)
3. One or two concrete next steps the user should try (different hardware, real workload instead of synthetic, quantization, speculative decoding, etc.)

Be direct and specific. No bullet points. No markdown headers. Just plain prose."""

            return self.llm(prompt).strip()
        except Exception as exc:
            logger.debug("Post-run analysis failed: %s", exc)
            return ""

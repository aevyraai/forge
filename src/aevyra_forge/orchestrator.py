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
    from aevyra_forge.runner import VLLMRunner


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
    llm_tokens: int = 0


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
    exploration_interval: int = 4  # reset to global best every N experiments
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
        from aevyra_forge.recipe import Recipe

        self._run_start = time.time()
        self._runner = None
        self._runner_args: list[str] = []
        current_layer = "config"

        # Build baseline recipe — pass workload's min context so max_model_len
        # is never capped below what the requests actually need.
        baseline_cfg = config_mod.baseline_config(
            self.hardware,
            model_name=self.model,
            min_context_len=self.workload.min_context_tokens(),
        )
        baseline_recipe = Recipe(
            model=self.model,
            hardware=self.hardware,
            config=baseline_cfg,
        )

        try:
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

                # Periodic re-exploration: reset to global best every N experiments
                # so the agent can explore branches it couldn't reach from the
                # current greedy path.
                exp_n = len(history)
                if (
                    self.cfg.exploration_interval > 0
                    and exp_n > 0
                    and exp_n % self.cfg.exploration_interval == 0
                ):
                    best_exp = self.store.best()
                    if best_exp and best_exp.recipe.id != current_recipe.id:
                        logger.info(
                            "forge │  ↩ re-exploring from best recipe %s "
                            "(score=%.1f) — interval=%d",
                            best_exp.recipe.id,
                            best_score,
                            self.cfg.exploration_interval,
                        )
                        current_recipe = best_exp.recipe

                # Ask agent for next mutation
                logger.info(
                    "forge │  experiment %d/%d — asking agent...",
                    exp_n,
                    self.cfg.max_experiments,
                )
                _tokens_before = getattr(self.llm, "tokens_used", 0)
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
                _exp_tokens = getattr(self.llm, "tokens_used", 0) - _tokens_before
                logger.info("forge │  tokens    : %d", _exp_tokens)

                # Empty changes = agent says converged
                if not decision.mutation.get("changes"):
                    logger.info("forge │  agent returned empty mutation — converged.")
                    break

                logger.info(
                    "forge ┌─ experiment %d/%d",
                    exp_n,
                    self.cfg.max_experiments,
                )
                logger.info("forge │  rationale : %s", decision.rationale)
                logger.info("forge │  mutation  : %s", decision.mutation.get("changes", {}))

                # Reject mutations that exactly match a previous CRASH or FAIL attempt
                proposed_changes = decision.mutation.get("changes", {})
                _already_tried = self._find_duplicate(history, proposed_changes, current_recipe)
                if _already_tried:
                    logger.warning(
                        "forge │  skipping duplicate mutation %s — already tried as exp %s (%s)",
                        proposed_changes,
                        _already_tried.id,
                        _already_tried.bench_result.status if _already_tried.bench_result else "?",
                    )
                    exp = Experiment(
                        id=f"dup-{len(history)}",
                        recipe=current_recipe,
                        agent_decision=decision,
                        score=0.0,
                        kept=False,
                    )
                    self.store.append(exp)
                    history = self.store.history()
                    continue

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
                exp.llm_tokens = _exp_tokens

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
                        "forge └─ ✓ KEPT      score=%.1f tok/s  p99=%.0fms  +%.1f%%  duration=%.0fs  llm=%d tok",
                        exp.score,
                        p99,
                        gain,
                        dur,
                        exp.llm_tokens,
                    )
                else:
                    exp.kept = False
                    logger.info(
                        "forge └─ ✗ REVERTED  score=%.1f tok/s  p99=%.0fms  best=%.1f  duration=%.0fs  llm=%d tok",
                        exp.score or 0.0,
                        p99,
                        best_score,
                        dur,
                        exp.llm_tokens,
                    )

                self.store.append(exp)
                history = self.store.history()
                self._print_table(history)

        finally:
            self._stop_runner()

        best_exp = self.store.best()
        best_recipe = best_exp.recipe if best_exp else baseline_recipe
        baseline_score = history[0].score or 0.0
        total_gain = 100 * (best_score - baseline_score) / max(baseline_score, 1e-9)
        elapsed = time.time() - self._run_start
        kept_count = sum(1 for e in history if e.kept)
        total_llm_tokens = sum(e.llm_tokens for e in history)
        logger.info("forge ══════════════════════════════════════════")
        logger.info("forge   total experiments : %d", len(history))
        logger.info("forge   kept              : %d", kept_count)
        logger.info("forge   baseline score    : %.1f tok/s", baseline_score)
        logger.info("forge   best score        : %.1f tok/s  (+%.1f%%)", best_score, total_gain)
        logger.info("forge   wall time         : %.0f min", elapsed / 60)
        logger.info("forge   llm tokens used   : %d", total_llm_tokens)
        logger.info("forge   best recipe gen   : %d  id=%s", best_recipe.generation, best_recipe.id)
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
        self._runner = None
        self._runner_args = []
        current_recipe = best_exp.recipe
        best_score = best_exp.score or 0.0
        current_layer = "config"

        logger.info(
            "Resuming from experiment %s with best_score=%.4f (%d experiments in history)",
            best_exp.id,
            best_score,
            len(history),
        )

        try:
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

        finally:
            self._stop_runner()

        best_exp = self.store.best()
        best_recipe = best_exp.recipe if best_exp else current_recipe
        return best_recipe, history

    def _ensure_runner(self, recipe: Recipe) -> "VLLMRunner":
        """Return a healthy VLLMRunner for *recipe*, restarting only when necessary.

        Restart is triggered when:
        - No server is currently running.
        - The vLLM argv for the new recipe differs from the running one.
        - The running process has died (``is_healthy()`` returns False).

        In dry-run mode a lightweight stub runner is always returned; the
        server is never actually started.
        """
        from aevyra_forge.runner import VLLMRunner, build_vllm_args

        new_args = build_vllm_args(recipe) if not self.cfg.dry_run else []

        runner: VLLMRunner | None = getattr(self, "_runner", None)
        current_args: list[str] = getattr(self, "_runner_args", [])

        need_restart = (
            runner is None
            or new_args != current_args
            or (not self.cfg.dry_run and not runner.is_healthy())
        )

        if need_restart:
            if runner is not None:
                logger.info("forge │  vLLM restarting (config changed)")
                runner.stop()
            else:
                logger.info("forge │  vLLM starting")
            runner = VLLMRunner(
                recipe,
                self.cfg.work_dir,
                dry_run=self.cfg.dry_run,
            )
            runner.start()
            self._runner: VLLMRunner | None = runner
            self._runner_args: list[str] = new_args
        else:
            logger.info("forge │  vLLM reused (args unchanged)")

        return runner  # type: ignore[return-value]

    def _stop_runner(self) -> None:
        """Stop the persistent runner if one is running."""
        runner = getattr(self, "_runner", None)
        if runner is not None:
            runner.stop()
            self._runner = None
            self._runner_args = []

    def _run_one(
        self,
        recipe: Recipe,
        *,
        agent_decision: "AgentDecision | None",
    ) -> Experiment:
        """Ensure vLLM is running for *recipe*, benchmark, score.

        Never raises — failures are captured in BenchResult.
        The server is kept alive after this call; ``_stop_runner`` tears it
        down at the end of the full run.
        """
        import time

        from aevyra_forge.bench import benchmark, score as bench_score, warmup

        t_start = time.time()
        exp = Experiment(
            id=recipe.id,
            recipe=recipe,
            agent_decision=agent_decision,
            started_at=t_start,
        )

        try:
            runner = self._ensure_runner(recipe)
            if not self.cfg.dry_run:
                warmup(server_url=runner.url(), workload=self.workload)
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
            error_str = str(exc)
            logger.error("Runner/bench failed for recipe %s: %s", recipe.id, error_str)
            # Stop the failed runner so log file handles and tee thread are cleaned up,
            # then mark it dead so the next experiment forces a fresh start.
            old_runner = getattr(self, "_runner", None)
            if old_runner is not None:
                try:
                    old_runner.stop()
                except Exception:
                    pass
            self._runner = None
            self._runner_args = []

            # vLLM suggests a safe max_model_len in the OOM message — apply it
            # so the NEXT experiment starts with a working context length.
            import re as _re
            _m = _re.search(r"estimated maximum model length is (\d+)", error_str)
            if _m:
                suggested = int(_m.group(1))
                min_ctx = self.workload.min_context_tokens()
                if suggested >= min_ctx:
                    logger.info(
                        "forge │  applying vLLM-suggested max_model_len=%d to current recipe",
                        suggested,
                    )
                    import copy as _copy
                    new_cfg = _copy.deepcopy(recipe.config)
                    new_cfg.max_model_len = suggested
                    # Update the running recipe so subsequent experiments inherit it
                    recipe = _copy.deepcopy(recipe)
                    recipe.config = new_cfg
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
            e
            for e in history
            if e.agent_decision and e.agent_decision.mutation.get("layer") == current_layer
        ][-self.cfg.convergence_window :]

        if len(recent) >= self.cfg.convergence_window and not any(e.kept for e in recent):
            next_layer = _layer_order[idx + 1]
            logger.info(
                "Layer %s saturated (%d consecutive non-improvements) — escalating to %s",
                current_layer,
                self.cfg.convergence_window,
                next_layer,
            )
            return next_layer

        return current_layer

    def _is_converged(self, history: list[Experiment]) -> bool:
        """True iff the last ``convergence_window`` experiments haven't
        improved the score by ``min_improvement_pct``.
        """
        if len(history) < self.cfg.convergence_window:
            return False

        recent = history[-self.cfg.convergence_window :]
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
                self.cfg.convergence_window,
                improvement,
                self.cfg.min_improvement_pct,
            )
            return True

        return False

    def _budget_exhausted(self, history: list[Experiment]) -> bool:
        """Check experiments, wall-clock, and dollar budgets."""
        import time

        if len(history) >= self.cfg.max_experiments:
            logger.info(
                "Budget exhausted: %d experiments reached (max=%d)",
                len(history),
                self.cfg.max_experiments,
            )
            return True

        run_start = getattr(self, "_run_start", None)
        if run_start is not None:
            elapsed_hours = (time.time() - run_start) / 3600
            if elapsed_hours >= self.cfg.max_wall_clock_hours:
                logger.info(
                    "Budget exhausted: %.2f hours elapsed (max=%.1f)",
                    elapsed_hours,
                    self.cfg.max_wall_clock_hours,
                )
                return True

        if self.cfg.max_dollars is not None:
            tokens_used = getattr(self.llm, "tokens_used", 0)
            # Rough estimate: $3/M input tokens (Sonnet pricing)
            estimated_cost = tokens_used / 1_000_000 * 3.0
            if estimated_cost >= self.cfg.max_dollars:
                logger.info(
                    "Budget exhausted: estimated cost $%.4f >= $%.2f limit",
                    estimated_cost,
                    self.cfg.max_dollars,
                )
                return True

        return False

    def _find_duplicate(
        self,
        history: list[Experiment],
        proposed_changes: dict,
        current_recipe: "Recipe",
    ) -> "Experiment | None":
        """Return a previous experiment that makes re-running proposed_changes pointless.

        Two levels of blocking:

        1. **CRASH / FAIL anywhere** — a mutation that crashes is bad
           regardless of the base recipe (OOM, bad flag, etc.).  Block it.

        2. **REVERTED from the same base recipe** — if we already tried
           max_num_seqs=32 on top of prefix_caching=True and it was worse,
           trying it again from the same base is a waste.  But if the base
           recipe has since changed (e.g. prefix_caching was just enabled),
           the same knob value might behave differently and should be allowed.

        A "same base recipe" match requires that the current recipe's values
        for the proposed fields are identical to the base recipe at the time
        of the past experiment.
        """
        if not proposed_changes:
            return None

        import dataclasses as _dc

        current_cfg = _dc.asdict(current_recipe.config)

        for exp in reversed(history):  # most recent first
            if not exp.agent_decision:
                continue
            past_changes = exp.agent_decision.mutation.get("changes", {})
            if not all(past_changes.get(k) == v for k, v in proposed_changes.items()):
                continue

            br = exp.bench_result

            # Level 1: always block crash/fail — bad regardless of base recipe
            if br and br.status in ("CRASH", "FAIL"):
                return exp
            if exp.score == 0.0 and not exp.kept:
                return exp

            # Level 2: block REVERTED only if the base recipe is the same
            # i.e. the fields being mutated had the same values in the current
            # recipe as they did when this past experiment was run
            if not exp.kept and br and br.status == "PASS":
                past_base_cfg = _dc.asdict(exp.recipe.config)
                # The base values for the proposed fields = past recipe MINUS the changes
                # (the past recipe IS the result of the mutation, so un-apply it)
                same_base = all(
                    past_base_cfg.get(k) == current_cfg.get(k)
                    for k in proposed_changes
                    if k not in past_changes  # fields not changed = base values
                ) and all(
                    # For the changed fields, the base value was current_cfg[k] == what
                    # it was BEFORE the past mutation — check via parent recipe if available
                    True  # conservative: only block if the exact same mutation on same parent
                    for k in proposed_changes
                )
                # Simpler and more robust: check if the current recipe's id matches
                # the parent of the past experiment
                if exp.recipe.parent_id == current_recipe.id:
                    return exp

        return None

    def _print_table(self, history: list[Experiment]) -> None:
        """Print a compact results table to stdout after each experiment."""
        header = f"{'#':>3}  {'score':>8}  {'tok/s':>8}  {'p99ms':>6}  {'kept':>4}  {'mutation'}"
        rows = [header, "─" * 70]
        for i, e in enumerate(history):
            br = e.bench_result
            score_s = f"{e.score:.1f}" if e.score is not None else "-"
            tps_s = f"{br.throughput_tokens_per_sec:.1f}" if br else "-"
            p99_s = f"{br.p99_latency_ms:.0f}" if br else "-"
            kept_s = "✓" if e.kept else "✗"
            if e.agent_decision:
                changes = e.agent_decision.mutation.get("changes", {})
                mut_s = ", ".join(f"{k}={v}" for k, v in changes.items())[:40]
            else:
                mut_s = "baseline"
            rows.append(f"{i:>3}  {score_s:>8}  {tps_s:>8}  {p99_s:>6}  {kept_s:>4}  {mut_s}")
        print("\n" + "\n".join(rows) + "\n")

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

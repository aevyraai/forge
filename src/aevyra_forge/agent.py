# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""The decision agent — single LLM call, reads history + playbook, proposes a mutation.

See AGENT.md → "Key concepts → The agent".

The agent is intentionally simple: one LLM call per experiment.
Reasoning lives in the playbook, not in agent loops. The orchestrator
owns scheduling, layer escalation, and budget — the agent only owns
"given this state, what should we try next."

Output is structured JSON. The orchestrator validates it against
``config.search_space`` / ``quant.search_space`` / ``kernel.search_space``
before applying.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aevyra_forge.llm import LLMFn
from aevyra_forge.playbook import Playbook
from aevyra_forge.recipe import Recipe
from aevyra_forge.workload import Workload


if TYPE_CHECKING:
    from aevyra_forge.orchestrator import Experiment


_AGENT_PROMPT = """\
You are the decision agent for Forge — an autonomous LLM serving autotuner.

# Workload summary
{workload_summary}

# Hardware
{hardware}

# Memory budget (computed for this hardware + model)
{memory_budget}

# Current recipe
{current_recipe_yaml}

# Recent experiments (last {history_window})
{history_text}

# Search space (legal values for this hardware + model)
{search_space_text}

# Playbook
{playbook_text}

# Your task
Propose ONE mutation to try next. Return STRICT JSON with these keys
and no other commentary:

{{
  "rationale": "<2-4 sentences explaining why this mutation should help>",
  "mutation": {{
    "layer": "config" | "quant" | "kernel",
    "changes": {{ "<field>": <value>, ... }}
  }},
  "expected_throughput_delta_pct": <number>,
  "expected_accuracy_delta": <number>
}}

Constraints:
- ``changes`` must reference fields legal at the chosen layer.
- Values must be inside the playbook's stated ranges.
- Do not retry a (field, value) pair that already appeared in a CRASH or FAIL experiment.
- Memory relationships (CRITICAL — read before touching memory knobs):
    * KV cache budget = gpu_memory_utilization × total_vram − model_weights
    * max_num_seqs and max_num_batched_tokens CONSUME KV cache — raising them
      requires MORE budget, not less.
    * Lowering gpu_memory_utilization SHRINKS the KV cache budget — it does NOT
      free space for more sequences; it makes OOM more likely.
    * If an experiment crashed with an OOM (error contains "KV cache" or
      "larger than the available"), the fix is to REDUCE max_num_seqs or
      INCREASE gpu_memory_utilization, never both at once.
    * Never pair max_num_seqs > baseline with gpu_memory_utilization < baseline.
- If you believe the search has converged, return mutation.changes = {{}}.
"""


@dataclass
class AgentDecision:
    rationale: str
    mutation: dict[str, Any]
    expected_throughput_delta_pct: float | None
    expected_accuracy_delta: float | None
    raw_response: str


class AgentError(Exception):
    """Raised when the agent's response can't be parsed or validated."""


def propose_next_experiment(
    *,
    history: list[Experiment],
    playbook: Playbook,
    current_recipe: Recipe,
    workload: Workload,
    llm: LLMFn,
    history_window: int = 10,
) -> AgentDecision:
    """Single LLM call. Reads state, returns a structured mutation.

    The raw LLM response is preserved so Origin can attribute failures
    back to specific agent statements.
    """
    from aevyra_forge import config as config_mod
    from aevyra_forge.playbook import format_for_agent

    recent = history[-history_window:] if len(history) > history_window else history
    history_lines: list[str] = []
    for exp in recent:
        score_str = f"{exp.score:.4f}" if exp.score is not None else "n/a"
        br = exp.bench_result
        status = br.status if br else "PENDING"
        kept = "KEPT" if exp.kept else "REVERTED"
        changes = ""
        if exp.agent_decision and exp.agent_decision.mutation.get("changes"):
            changes = str(exp.agent_decision.mutation["changes"])
        rationale = exp.agent_decision.rationale[:80] if exp.agent_decision else ""
        # Mark duplicate-skipped or search-space-rejected experiments clearly
        # so the agent knows not to propose these values again.
        if exp.id.startswith("dup-"):
            line = (
                f"  [{exp.id}] DUPLICATE SKIPPED — changes={changes} were already "
                f"tried and failed. Do not propose this again."
            )
        elif exp.id.startswith("rej-"):
            line = (
                f"  [{exp.id}] SEARCH SPACE VIOLATION — changes={changes} are ILLEGAL "
                f"for this hardware+model (value not in the search space listed above). "
                f"Do not propose these values again under any circumstances."
            )
        else:
            line = (
                f"  [{exp.id}] gen={exp.recipe.generation} score={score_str} "
                f"status={status} {kept} changes={changes} rationale={rationale!r}"
            )
            # Surface the failure reason so the agent doesn't repeat OOM / bad-flag crashes
            if br and br.error and status in ("CRASH", "FAIL"):
                error_snippet = br.error.replace("\n", " ")[:200]
                line += f" error={error_snippet!r}"
        history_lines.append(line)
    history_text = "\n".join(history_lines) if history_lines else "  (no experiments yet)"
    playbook_text = format_for_agent(playbook, current_recipe.hardware, layer="config")

    # Compute memory budget and legal search space for this hardware + model
    quant_method = current_recipe.quant.method if current_recipe.quant else "bf16"
    hw = current_recipe.hardware
    weight_gb = config_mod.estimate_weight_gb(current_recipe.model, quant_method)
    budget_90 = config_mod.kv_cache_budget_gb(hw, weight_gb, 0.90)
    budget_92 = config_mod.kv_cache_budget_gb(hw, weight_gb, 0.92)
    computed_safe_seqs = config_mod._safe_max_num_seqs(hw, current_recipe.model, quant_method)
    # If the current recipe is already running at a higher value without OOM,
    # the formula is too conservative — trust the observed working value.
    actual_seqs = current_recipe.config.max_num_seqs
    safe_seqs = max(computed_safe_seqs, actual_seqs)
    space = config_mod.search_space(hw, current_recipe.model, quant_method)
    # Ensure the search space reflects the observed working ceiling
    if safe_seqs not in space.get("max_num_seqs", []):
        space["max_num_seqs"] = sorted(set(space.get("max_num_seqs", [])) | {safe_seqs})

    kv_cfg = config_mod.fetch_model_kv_config(current_recipe.model)
    if kv_cfg:
        kv_source = (
            f"exact (n_layers={kv_cfg['num_hidden_layers']}, "
            f"n_kv_heads={kv_cfg['num_key_value_heads']}, "
            f"head_dim={kv_cfg['head_dim']}, "
            f"max_pos={kv_cfg['max_position_embeddings']})"
        )
    else:
        kv_source = "estimated from params_b (HuggingFace config unavailable)"

    memory_budget = (
        f"GPU VRAM        : {hw.memory_gb_per_gpu} GB × {hw.count} = {hw.memory_gb_per_gpu * hw.count} GB total\n"
        f"Model weights   : ~{weight_gb:.1f} GB ({quant_method})\n"
        f"KV calc source  : {kv_source}\n"
        f"KV cache budget at gpu_memory_utilization=0.90 : {budget_90:.1f} GB\n"
        f"KV cache budget at gpu_memory_utilization=0.92 : {budget_92:.1f} GB\n"
        f"Formula ceiling for max_num_seqs : {computed_safe_seqs}\n"
        f"Observed working max_num_seqs    : {actual_seqs} (current recipe runs without OOM)\n"
        f"Effective ceiling                : {safe_seqs}\n"
        f"RULE: max_num_seqs must stay in [{actual_seqs}, {safe_seqs}]. "
        f"Reducing below {actual_seqs} has already been tried or is pointless — pick a different knob."
    )

    search_space_text = "\n".join(
        f"  {k}: {v}"
        for k, v in sorted(space.items())
        if k
        in (
            "max_num_seqs",
            "max_num_batched_tokens",
            "block_size",
            "gpu_memory_utilization",
            "kv_cache_dtype",
            "attention_backend",
        )
    )

    prompt = _AGENT_PROMPT.format(
        workload_summary=workload.summary(),
        hardware=current_recipe.hardware.label(),
        memory_budget=memory_budget,
        current_recipe_yaml=current_recipe.to_yaml(),
        history_window=len(recent),
        history_text=history_text,
        search_space_text=search_space_text,
        playbook_text=playbook_text,
    )

    raw = llm(prompt)

    import json
    import re

    def _parse(text: str) -> dict:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        match = re.search(r"\{[\s\S]+\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise AgentError(f"Could not parse JSON from agent response. Response was:\n{text[:500]}")

    parsed = _parse(raw)
    for key in ("rationale", "mutation"):
        if key not in parsed:
            raise AgentError(f"Agent response missing required key {key!r}. Parsed: {parsed}")

    return AgentDecision(
        rationale=str(parsed.get("rationale", "")),
        mutation=parsed.get("mutation", {}),
        expected_throughput_delta_pct=parsed.get("expected_throughput_delta_pct"),
        expected_accuracy_delta=parsed.get("expected_accuracy_delta"),
        raw_response=raw,
    )

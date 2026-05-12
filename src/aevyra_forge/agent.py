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

# Current recipe
{current_recipe_yaml}

# Recent experiments (last {history_window})
{history_text}

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
    from aevyra_forge.playbook import format_for_agent

    recent = history[-history_window:] if len(history) > history_window else history
    history_lines: list[str] = []
    for exp in recent:
        score_str = f"{exp.score:.4f}" if exp.score is not None else "n/a"
        status = exp.bench_result.status if exp.bench_result else "PENDING"
        kept = "KEPT" if exp.kept else "REVERTED"
        changes = ""
        if exp.agent_decision and exp.agent_decision.mutation.get("changes"):
            changes = str(exp.agent_decision.mutation["changes"])
        rationale = exp.agent_decision.rationale[:80] if exp.agent_decision else ""
        history_lines.append(
            f"  [{exp.id}] gen={exp.recipe.generation} score={score_str} "
            f"status={status} {kept} changes={changes} rationale={rationale!r}"
        )
    history_text = "\n".join(history_lines) if history_lines else "  (no experiments yet)"
    playbook_text = format_for_agent(playbook, current_recipe.hardware, layer="config")

    prompt = _AGENT_PROMPT.format(
        workload_summary=workload.summary(),
        hardware=current_recipe.hardware.label(),
        current_recipe_yaml=current_recipe.to_yaml(),
        history_window=len(recent),
        history_text=history_text,
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

# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Layer 2 — quantization recipe selection.

See AGENT.md → "The three layers" and "v0 scope".

**Not implemented in v0.** This module exists so the architecture is
visible from day one — the orchestrator's layer-escalation logic
calls ``quant.mutate`` once Layer 1 saturates; it must exist as a
named symbol even when it raises.

v0.2 scope:

- Per-vendor quant method availability (FP8 E4M3 on Hopper+ only;
  OCP FP8 on MI300X+; INT4 AWQ via llm-compressor on both).
- Per-layer bit-width hints (the playbook says which layers tolerate
  what — outliers stay higher precision).
- Calibration dataset selection (workload-shape-conditioned).
- Re-quantization cost model (a quant change is much more expensive
  than a config change — the orchestrator must budget accordingly).
"""

from __future__ import annotations

from typing import Any

from aevyra_forge.recipe import HardwareSpec, QuantRecipe, Recipe


def search_space(hardware: HardwareSpec) -> dict[str, list[Any]]:
    """Return legal quant choices for this hardware. NOT IMPLEMENTED in v0."""
    raise NotImplementedError("Layer 2 (quantization) is v0.2 scope. See AGENT.md.")


def mutate(recipe: Recipe, mutation: dict[str, Any]) -> Recipe:
    """Apply a quant mutation. NOT IMPLEMENTED in v0."""
    raise NotImplementedError("Layer 2 (quantization) is v0.2 scope. See AGENT.md.")


def baseline_quant(hardware: HardwareSpec, model: str) -> QuantRecipe:
    """The starting-point quant for this (hardware, model). NOT IMPLEMENTED in v0."""
    raise NotImplementedError("Layer 2 (quantization) is v0.2 scope. See AGENT.md.")


def estimated_quant_cost_s(recipe: Recipe, target_quant: QuantRecipe) -> float:
    """How many seconds the orchestrator should budget for a re-quantization.

    Used by the layer-escalation logic. INT4 calibration can take 30+ min;
    FP8 dynamic is near-zero. The orchestrator needs to know before
    deciding to escalate.

    NOT IMPLEMENTED in v0. Returns a placeholder for now.
    """
    raise NotImplementedError("Layer 2 (quantization) is v0.2 scope. See AGENT.md.")

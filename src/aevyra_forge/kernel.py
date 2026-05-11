# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Layer 3 — custom kernel synthesis hook (AutoKernel integration).

See AGENT.md → "The three layers" and "v0 scope".

**Not implemented in v0.3+ scope.** This module is the integration
seam between Forge and AutoKernel. The orchestrator escalates here
only when Layer 1 and Layer 2 have saturated and the workload is
exotic enough that standard kernels leave performance on the table.

When implemented, this module will:

- Profile the deployed model with the current recipe (using vLLM's
  profiler) to find hot kernels.
- Hand the hottest kernel + the model's specific shapes to
  AutoKernel.
- Take AutoKernel's output (a verified faster Triton/CUDA kernel)
  and produce a ``KernelOverride`` that vLLM can load.
- Track the kernel's provenance back to the AutoKernel run that
  produced it.

The escalation gate is intentional: kernel synthesis is the most
expensive layer (~30-90 min per experiment) and only pays off on
non-standard shapes. The playbook decides when to escalate.
"""

from __future__ import annotations

from typing import Any

from aevyra_forge.recipe import KernelOverride, Recipe


def search_space() -> dict[str, list[Any]]:
    """Which kernels are candidates for synthesis. NOT IMPLEMENTED in v0."""
    raise NotImplementedError("Layer 3 (kernel synthesis) is v0.3 scope. See AGENT.md.")


def profile_hot_kernels(recipe: Recipe) -> list[str]:
    """Find the kernels burning the most time on this recipe.

    NOT IMPLEMENTED in v0. Returns the list of op names AutoKernel
    should target (e.g. ["attention", "fused_mlp"]).
    """
    raise NotImplementedError("Layer 3 (kernel synthesis) is v0.3 scope. See AGENT.md.")


def synthesize_kernel(op_name: str, recipe: Recipe) -> KernelOverride:
    """Hand control to AutoKernel for the named op; return the produced override.

    NOT IMPLEMENTED in v0. This is the integration seam — calls into
    ``autokernel`` package once it's installable.
    """
    raise NotImplementedError("Layer 3 (kernel synthesis) is v0.3 scope. See AGENT.md.")

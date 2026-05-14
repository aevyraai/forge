# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""aevyra-forge — autonomous overnight optimizer for LLM inference deployments.

Quick start::

    from aevyra_forge import Forge, HardwareSpec, load_playbook, workload_synthetic
    from aevyra_forge.llm import anthropic_llm

    forge = Forge(
        model="meta-llama/Llama-3.1-8B-Instruct",
        hardware=HardwareSpec(vendor="nvidia", gpu_type="A100", count=1, memory_gb_per_gpu=80),
        workload=workload_synthetic(n_requests=1000),
        playbook=load_playbook("playbooks/default.md"),
        llm=anthropic_llm(),
    )
    best_recipe, history = forge.run()

See AGENT.md for the full architecture spec.
"""

from __future__ import annotations

from aevyra_forge.bench import BenchResult
from aevyra_forge.orchestrator import Experiment, ForgeConfig, Orchestrator
from aevyra_forge.playbook import Playbook, load_playbook
from aevyra_forge.recipe import (
    HardwareSpec,
    KernelOverride,
    QuantRecipe,
    Recipe,
    VLLMConfig,
)
from aevyra_forge.result import ForgeRun, ForgeStore
from aevyra_forge.workload import (
    Workload,
    WorkloadRequest,
    workload_from_jsonl,
    workload_synthetic,
)

__version__ = "0.1.0"


class ForgeError(Exception):
    """Base exception for Forge."""


# `Forge` is a convenience facade that wires Orchestrator + Runner together.
# Sonnet: implement this in orchestrator.py (or here) once the modules underneath
# are in place. See AGENT.md "v0 scope" item 7.
Forge = Orchestrator


__all__ = [
    "BenchResult",
    "Experiment",
    "Forge",
    "ForgeConfig",
    "ForgeError",
    "ForgeRun",
    "ForgeStore",
    "HardwareSpec",
    "KernelOverride",
    "Orchestrator",
    "Playbook",
    "QuantRecipe",
    "Recipe",
    "VLLMConfig",
    "Workload",
    "WorkloadRequest",
    "__version__",
    "load_playbook",
    "workload_from_jsonl",
    "workload_synthetic",
]

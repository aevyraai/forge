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

"""Recipe dataclasses — the single artifact the orchestrator mutates.

See AGENT.md → "Module-by-module spec → recipe.py" for the full schema.
This module owns:

- ``HardwareSpec``      — vendor + GPU type + count + memory
- ``VLLMConfig``        — Layer 1 search space (vLLM serving args)
- ``QuantRecipe``       — Layer 2 search space (quant + KV cache precision)
- ``KernelOverride``    — Layer 3 search space (custom kernel hooks)
- ``Recipe``            — the full bundle, YAML-serializable

Sonnet: fill in ``to_yaml`` / ``from_yaml`` using PyYAML. Add fields
to ``VLLMConfig`` as the default playbook references them. Keep
defaults equal to vLLM's defaults so an empty Recipe runs vLLM out-of-
the-box.
"""

from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import yaml


@dataclass
class HardwareSpec:
    vendor: Literal["nvidia", "amd", "intel", "google", "other"]
    gpu_type: str
    count: int
    memory_gb_per_gpu: int

    def label(self) -> str:
        return f"{self.vendor}/{self.gpu_type}x{self.count}"


@dataclass
class VLLMConfig:
    """Layer 1 search space. All fields tunable; defaults = vLLM defaults.

    Add new fields as the default playbook references them. Keep this
    flat (no nested dicts) so JSON/YAML diffs stay reviewable.
    """

    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    block_size: int = 16
    gpu_memory_utilization: float = 0.9
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = True
    swap_space: int = 4
    kv_cache_dtype: Literal["auto", "fp8", "fp16", "bf16"] = "auto"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    speculative_model: str | None = None
    num_speculative_tokens: int = 0
    attention_backend: str | None = None  # vendor-specific (FA3, ROCM_AITER_FA, ...)


@dataclass
class QuantRecipe:
    """Layer 2 search space. v0: defaults only — mutating raises NotImplementedError."""

    method: Literal["fp16", "bf16", "int4_awq", "int4_gptq", "fp8_e4m3", "int8"] = "bf16"
    kv_cache_quant: Literal["none", "fp8", "int8"] = "none"
    calibration_dataset: str | None = None
    per_layer_overrides: dict[str, str] = field(default_factory=dict)


@dataclass
class KernelOverride:
    """Layer 3 search space. v0: never populated."""

    op_name: str
    kernel_source_path: str


@dataclass
class Recipe:
    """The full deployment recipe — model + hardware + all three layers."""

    model: str
    hardware: HardwareSpec
    config: VLLMConfig = field(default_factory=VLLMConfig)
    quant: QuantRecipe = field(default_factory=QuantRecipe)
    kernels: list[KernelOverride] = field(default_factory=list)
    parent_id: str | None = None
    generation: int = 0
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Recipe:
        hw = HardwareSpec(**d["hardware"])
        cfg = VLLMConfig(**d["config"])
        quant = QuantRecipe(**d["quant"])
        kernels = [KernelOverride(**k) for k in d.get("kernels", [])]
        return cls(
            model=d["model"],
            hardware=hw,
            config=cfg,
            quant=quant,
            kernels=kernels,
            parent_id=d.get("parent_id"),
            generation=d.get("generation", 0),
            id=d.get("id", str(uuid.uuid4())[:8]),
        )

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(cls, s: str) -> Recipe:
        return cls.from_dict(yaml.safe_load(s))

    def diff(self, other: Recipe) -> dict[str, Any]:
        """Return only the VLLMConfig fields that differ between self and other."""
        a = dataclasses.asdict(self.config)
        b = dataclasses.asdict(other.config)
        return {k: {"from": a[k], "to": b[k]} for k in a if a[k] != b[k]}

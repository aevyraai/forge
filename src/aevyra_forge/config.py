# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Layer 1 — vLLM config search space + mutation operators.

See AGENT.md → "Module-by-module spec → config.py".

This module owns:

- The *legal* values of each ``VLLMConfig`` field on each hardware type.
- The mutation function that the agent's JSON output drives.

The agent picks the mutations; this module validates them and applies
them. Anything the agent proposes that's outside the search space is
rejected with a clear error (logged and surfaced as an agent failure
that Origin can attribute).
"""

from __future__ import annotations

from typing import Any

from aevyra_forge.recipe import HardwareSpec, Recipe, VLLMConfig


def search_space(hardware: HardwareSpec) -> dict[str, list[Any]]:
    """Return legal values per VLLMConfig field for this hardware."""
    base: dict[str, list[Any]] = {
        "max_num_seqs": [16, 32, 64, 128, 256, 512, 1024],
        "max_num_batched_tokens": [2048, 4096, 8192, 16384, 32768],
        "block_size": [16, 32],
        "gpu_memory_utilization": [0.85, 0.88, 0.90, 0.92, 0.95],
        "enable_prefix_caching": [True, False],
        "enable_chunked_prefill": [True, False],
        "swap_space": [0, 4, 8, 16],
        "kv_cache_dtype": ["auto", "bf16"],
        "tensor_parallel_size": list(range(1, hardware.count + 1)),
        "pipeline_parallel_size": [1],
        "speculative_model": [None],
        "num_speculative_tokens": [0],
        "attention_backend": [None],
    }

    if hardware.vendor == "nvidia":
        gpu = hardware.gpu_type.upper()
        if any(g in gpu for g in ("H100", "H200", "B100", "B200")):
            base["kv_cache_dtype"] = ["auto", "fp8", "bf16"]
            base["attention_backend"] = [None, "flash-attn-3"]
        elif "A100" in gpu:
            base["attention_backend"] = [None, "flash-attn-2"]
        elif "T4" in gpu:
            base["max_num_seqs"] = [16, 32, 64]
            base["block_size"] = [16]

    elif hardware.vendor == "amd":
        base["attention_backend"] = [None, "ROCM_AITER_FA", "AITER_MLA"]
        base["block_size"] = [16, 32]
        if "MI300" in hardware.gpu_type.upper():
            base["kv_cache_dtype"] = ["auto", "fp8", "bf16"]

    return base


def mutate(recipe: Recipe, mutation: dict[str, Any]) -> Recipe:
    """Apply an agent-proposed mutation. Return a new Recipe."""
    import copy
    import dataclasses
    import uuid

    layer = mutation.get("layer", "config")
    changes = mutation.get("changes", {})

    if not changes:
        new = copy.deepcopy(recipe)
        new.parent_id = recipe.id
        new.generation = recipe.generation + 1
        new.id = str(uuid.uuid4())[:8]
        return new

    if layer != "config":
        raise NotImplementedError(f"Layer '{layer}' mutation is v0.2+")

    space = search_space(recipe.hardware)
    current = dataclasses.asdict(recipe.config)

    for field_name, value in changes.items():
        if field_name not in current:
            raise ValueError(
                f"Unknown VLLMConfig field: {field_name!r}. Known fields: {sorted(current.keys())}"
            )
        if field_name in space and value not in space[field_name]:
            raise ValueError(
                f"Value {value!r} for {field_name!r} is outside the search space. "
                f"Legal values: {space[field_name]}"
            )
        current[field_name] = value

    new_config = VLLMConfig(**current)
    return Recipe(
        model=recipe.model,
        hardware=recipe.hardware,
        config=new_config,
        quant=copy.deepcopy(recipe.quant),
        kernels=list(recipe.kernels),
        parent_id=recipe.id,
        generation=recipe.generation + 1,
        id=str(uuid.uuid4())[:8],
    )


def baseline_config(hardware: HardwareSpec) -> VLLMConfig:
    """Reasonable starting defaults for the given hardware."""
    cfg = VLLMConfig()
    gpu = hardware.gpu_type.upper()
    vendor = hardware.vendor

    if vendor == "nvidia":
        if any(g in gpu for g in ("H100", "H200", "B200")):
            cfg.max_num_seqs = 512
            cfg.max_num_batched_tokens = 16384
            cfg.block_size = 32
            cfg.enable_chunked_prefill = True
        elif "A100" in gpu:
            cfg.max_num_seqs = 256
            cfg.max_num_batched_tokens = 8192
        elif "T4" in gpu:
            cfg.max_num_seqs = 32
            cfg.max_num_batched_tokens = 2048
            cfg.gpu_memory_utilization = 0.88

    elif vendor == "amd":
        if "MI300" in gpu:
            cfg.max_num_seqs = 512
            cfg.max_num_batched_tokens = 16384
            cfg.block_size = 32
            cfg.attention_backend = "ROCM_AITER_FA"

    return cfg

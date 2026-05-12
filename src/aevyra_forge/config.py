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

import re
from typing import Any

from aevyra_forge.recipe import HardwareSpec, Recipe, VLLMConfig


# ---------------------------------------------------------------------------
# Memory budget helpers
# ---------------------------------------------------------------------------


def estimate_weight_gb(model_name: str, quant_method: str = "bf16") -> float:
    """Estimate model weight memory in GB from the model name and quant dtype.

    Parses parameter count from names like ``llama-3-8b``, ``qwen2.5-72b``,
    ``mistral-7b-instruct``. Falls back to 8 B if unparseable.

    Args:
        model_name: HuggingFace repo ID or local path.
        quant_method: One of ``bf16``, ``fp16``, ``fp8``, ``int8``,
            ``int4_awq``, ``int4_gptq``. Controls bytes-per-parameter.
    """
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?:\b|$)", model_name, re.IGNORECASE)
    params_b = float(m.group(1)) if m else 8.0

    bytes_per_param: dict[str, float] = {
        "fp32": 4.0,
        "fp16": 2.0,
        "bf16": 2.0,
        "fp8": 1.0,
        "fp8_e4m3": 1.0,
        "int8": 1.0,
        "int4_awq": 0.5,
        "int4_gptq": 0.5,
        "int4": 0.5,
    }
    bpp = bytes_per_param.get(quant_method.lower(), 2.0)
    # +10 % overhead for activations, buffers, CUDA context
    return params_b * bpp * 1.1


def kv_cache_budget_gb(
    hardware: HardwareSpec,
    weight_gb: float,
    gpu_memory_utilization: float = 0.90,
) -> float:
    """Available KV cache memory in GB given weights and utilization.

    With tensor parallelism, weights are sharded across GPUs, so the
    per-GPU weight footprint is ``weight_gb / hardware.count``.
    """
    weight_per_gpu = weight_gb / max(1, hardware.count)
    budget = gpu_memory_utilization * hardware.memory_gb_per_gpu - weight_per_gpu
    return max(0.0, budget)


def _safe_max_num_seqs(
    hardware: HardwareSpec,
    model_name: str,
    quant_method: str = "bf16",
) -> int:
    """Upper bound on ``max_num_seqs`` that avoids KV-cache OOM.

    Uses a conservative heuristic: compute KV budget at 0.92 utilization,
    then estimate sequences from params_b.

    KV memory per sequence scales with model depth/width. Empirical rule:
      kv_gb_per_32_seqs ≈ params_b × 0.05
    (Derived from: 8B → ~32 seqs costs ~0.4 GB; 70B → ~32 seqs costs ~3.5 GB)

    Returns the largest value in ``[8, 16, 32, 64, 128, 256, 512, 1024]``
    that fits, minimum 8.
    """
    weight_gb = estimate_weight_gb(model_name, quant_method)
    # Use 0.92 as the ceiling to compute the theoretical max
    budget = kv_cache_budget_gb(hardware, weight_gb, gpu_memory_utilization=0.92)
    if budget <= 0:
        return 8

    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?:\b|$)", model_name, re.IGNORECASE)
    params_b = float(m.group(1)) if m else 8.0
    kv_gb_per_32 = max(0.05, params_b * 0.05)
    max_safe = max(8, int(budget / kv_gb_per_32 * 32))

    candidates = [8, 16, 32, 64, 128, 256, 512, 1024]
    for v in reversed(candidates):
        if v <= max_safe:
            return v
    return 8


def _vram_tier(hardware: HardwareSpec) -> str:
    """Classify GPU(s) into a VRAM tier for search-space sizing.

    Uses total VRAM (memory_gb_per_gpu × count) so multi-GPU rigs
    get the right tier automatically.

    Returns one of: ``small``, ``medium``, ``large``, ``xlarge``.
    """
    total = hardware.memory_gb_per_gpu * hardware.count
    if total <= 20:
        return "small"    # T4 16 GB, V100 16 GB, RTX 3080 10 GB
    elif total <= 50:
        return "medium"   # A10G 24 GB, RTX 4090 24 GB, A40 48 GB
    elif total <= 100:
        return "large"    # A100 40/80 GB, H100 80 GB, 2×A10G 48 GB
    else:
        return "xlarge"   # MI300X 192 GB, H200, 4×A100, 8×A100


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------


def search_space(
    hardware: HardwareSpec,
    model_name: str = "",
    quant_method: str = "bf16",
) -> dict[str, list[Any]]:
    """Return legal values per VLLMConfig field for this hardware + model.

    Merges three sources of knowledge:
    1. ``_safe_max_num_seqs`` — memory-budget-derived upper bound.
    2. GPU name matching — vendor-specific capabilities (FP8, attention backends).
    3. VRAM tier — fallback for unknown GPU models.
    """
    gpu = hardware.gpu_type.upper()
    tier = _vram_tier(hardware)
    safe_seqs = _safe_max_num_seqs(hardware, model_name, quant_method) if model_name else None

    # Default max_num_seqs ladder by VRAM tier (without model knowledge)
    tier_seqs: dict[str, list[int]] = {
        "small":  [8, 16, 32],
        "medium": [16, 32, 64, 128],
        "large":  [32, 64, 128, 256, 512],
        "xlarge": [64, 128, 256, 512, 1024],
    }
    base_seqs = tier_seqs[tier]

    # Narrow further using computed memory budget when model is known
    all_candidates = [8, 16, 32, 64, 128, 256, 512, 1024]
    if safe_seqs is not None:
        seqs = [v for v in all_candidates if v <= safe_seqs] or [8]
    else:
        seqs = base_seqs

    base: dict[str, list[Any]] = {
        "max_num_seqs": seqs,
        "max_num_batched_tokens": [2048, 4096, 8192, 16384, 32768],
        "block_size": [16, 32] if tier != "small" else [16],
        "gpu_memory_utilization": [0.80, 0.85, 0.88, 0.90, 0.92, 0.95],
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

    # --- NVIDIA ---
    if hardware.vendor == "nvidia":
        if any(g in gpu for g in ("H100", "H200", "B100", "B200")):
            base["kv_cache_dtype"] = ["auto", "fp8", "bf16"]
            base["attention_backend"] = [None, "flash-attn-3"]
        elif "A100" in gpu:
            base["attention_backend"] = [None, "flash-attn-2"]
        elif any(g in gpu for g in ("T4", "V100")):
            # Low VRAM: hard cap block_size and utilization ceiling
            base["block_size"] = [16]
            # gpu_memory_utilization above 0.92 is unstable on T4
            base["gpu_memory_utilization"] = [0.80, 0.85, 0.88, 0.90, 0.92]
        elif "A10" in gpu:
            base["attention_backend"] = [None, "flash-attn-2"]

    # --- AMD ---
    elif hardware.vendor == "amd":
        base["attention_backend"] = [None, "ROCM_AITER_FA", "AITER_MLA", "ROCM_FLASH_ATTN"]
        if tier != "small":
            base["block_size"] = [16, 32]
        if "MI300" in gpu or "MI250" in gpu:
            base["kv_cache_dtype"] = ["auto", "fp8", "bf16"]

    return base


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------


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

    quant_method = recipe.quant.method if recipe.quant else "bf16"
    space = search_space(recipe.hardware, model_name=recipe.model, quant_method=quant_method)
    current = dataclasses.asdict(recipe.config)

    for field_name, value in changes.items():
        if field_name not in current:
            raise ValueError(
                f"Unknown VLLMConfig field: {field_name!r}. "
                f"Known fields: {sorted(current.keys())}"
            )
        if field_name in space and value not in space[field_name]:
            raise ValueError(
                f"Value {value!r} for {field_name!r} is outside the search space "
                f"for {recipe.hardware.label()} with model {recipe.model!r}. "
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


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def baseline_config(hardware: HardwareSpec, model_name: str = "") -> VLLMConfig:
    """Reasonable starting defaults for the given hardware + model.

    When ``model_name`` is provided the memory budget is used to pick
    a safe starting ``max_num_seqs``. Falls back to VRAM tier otherwise.
    """
    cfg = VLLMConfig()
    gpu = hardware.gpu_type.upper()
    vendor = hardware.vendor
    tier = _vram_tier(hardware)

    # --- Compute a safe baseline max_num_seqs ---
    if model_name:
        safe = _safe_max_num_seqs(hardware, model_name)
        # Use half the safe ceiling as a conservative baseline
        candidates = [8, 16, 32, 64, 128, 256, 512]
        baseline_seqs = 16
        for v in candidates:
            if v <= safe // 2:
                baseline_seqs = v
        cfg.max_num_seqs = baseline_seqs
    else:
        tier_defaults = {"small": 16, "medium": 64, "large": 256, "xlarge": 512}
        cfg.max_num_seqs = tier_defaults[tier]

    # --- max_num_batched_tokens: at least max_num_seqs × avg_input (assume ~128) ---
    cfg.max_num_batched_tokens = max(2048, cfg.max_num_seqs * 128)
    # Cap to 16384 for small/medium tiers to avoid excessive memory
    if tier in ("small", "medium"):
        cfg.max_num_batched_tokens = min(cfg.max_num_batched_tokens, 8192)

    # --- Vendor / GPU specific overrides ---
    if vendor == "nvidia":
        if any(g in gpu for g in ("H100", "H200", "B200")):
            cfg.block_size = 32
            cfg.enable_chunked_prefill = True
            cfg.gpu_memory_utilization = 0.90
        elif "A100" in gpu:
            cfg.gpu_memory_utilization = 0.90
        elif any(g in gpu for g in ("T4", "V100")):
            cfg.block_size = 16
            cfg.gpu_memory_utilization = 0.88
        elif "A10" in gpu:
            cfg.gpu_memory_utilization = 0.90

    elif vendor == "amd":
        if "MI300" in gpu or "MI250" in gpu:
            cfg.block_size = 32
            cfg.attention_backend = "ROCM_AITER_FA"
            cfg.gpu_memory_utilization = 0.90

    return cfg

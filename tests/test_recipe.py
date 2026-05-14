# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for aevyra_forge.recipe."""

import yaml

from aevyra_forge.recipe import (
    HardwareSpec,
    KernelOverride,
    QuantRecipe,
    Recipe,
    VLLMConfig,
)


# ---------------------------------------------------------------------------
# HardwareSpec
# ---------------------------------------------------------------------------


def test_hardware_spec_label() -> None:
    hw = HardwareSpec(vendor="nvidia", gpu_type="A100", count=4, memory_gb_per_gpu=80)
    assert hw.label() == "nvidia/A100x4"


def test_hardware_spec_label_amd() -> None:
    hw = HardwareSpec(vendor="amd", gpu_type="MI300X", count=8, memory_gb_per_gpu=192)
    assert hw.label() == "amd/MI300Xx8"


def test_hardware_spec_single_gpu() -> None:
    hw = HardwareSpec(vendor="nvidia", gpu_type="T4", count=1, memory_gb_per_gpu=16)
    assert hw.label() == "nvidia/T4x1"


# ---------------------------------------------------------------------------
# VLLMConfig defaults
# ---------------------------------------------------------------------------


def test_vllm_config_defaults() -> None:
    cfg = VLLMConfig()
    assert cfg.max_num_seqs == 256
    assert cfg.gpu_memory_utilization == 0.9
    assert cfg.enable_prefix_caching is False
    assert cfg.enable_chunked_prefill is True
    assert cfg.tensor_parallel_size == 1
    assert cfg.max_model_len is None


def test_vllm_config_custom() -> None:
    cfg = VLLMConfig(max_num_seqs=64, enable_prefix_caching=True, gpu_memory_utilization=0.85)
    assert cfg.max_num_seqs == 64
    assert cfg.enable_prefix_caching is True
    assert cfg.gpu_memory_utilization == 0.85


# ---------------------------------------------------------------------------
# QuantRecipe defaults
# ---------------------------------------------------------------------------


def test_quant_recipe_defaults() -> None:
    q = QuantRecipe()
    assert q.method == "bf16"
    assert q.kv_cache_quant == "none"
    assert q.calibration_dataset is None
    assert q.per_layer_overrides == {}


# ---------------------------------------------------------------------------
# Recipe round-trip: to_dict / from_dict
# ---------------------------------------------------------------------------


def _make_recipe(**kwargs: object) -> Recipe:
    hw = HardwareSpec(vendor="nvidia", gpu_type="T4", count=1, memory_gb_per_gpu=16)
    return Recipe(model="meta-llama/Llama-3.2-1B-Instruct", hardware=hw, **kwargs)  # type: ignore[arg-type]


def test_recipe_to_dict_has_expected_keys() -> None:
    r = _make_recipe()
    d = r.to_dict()
    assert "model" in d
    assert "hardware" in d
    assert "config" in d
    assert "quant" in d
    assert "kernels" in d
    assert "generation" in d
    assert "id" in d


def test_recipe_from_dict_roundtrip() -> None:
    r = _make_recipe()
    r.config.enable_prefix_caching = True
    r.config.max_num_seqs = 64
    d = r.to_dict()
    r2 = Recipe.from_dict(d)
    assert r2.model == r.model
    assert r2.hardware.gpu_type == r.hardware.gpu_type
    assert r2.config.enable_prefix_caching is True
    assert r2.config.max_num_seqs == 64
    assert r2.quant.method == r.quant.method
    assert r2.id == r.id
    assert r2.generation == r.generation


def test_recipe_from_dict_preserves_kernels() -> None:
    r = _make_recipe()
    r.kernels = [KernelOverride(op_name="flash_attn", kernel_source_path="/tmp/fa.cu")]
    d = r.to_dict()
    r2 = Recipe.from_dict(d)
    assert len(r2.kernels) == 1
    assert r2.kernels[0].op_name == "flash_attn"


# ---------------------------------------------------------------------------
# Recipe round-trip: to_yaml / from_yaml
# ---------------------------------------------------------------------------


def test_recipe_to_yaml_is_valid_yaml() -> None:
    r = _make_recipe()
    s = r.to_yaml()
    parsed = yaml.safe_load(s)
    assert parsed["model"] == r.model
    assert parsed["hardware"]["vendor"] == "nvidia"


def test_recipe_yaml_roundtrip() -> None:
    r = _make_recipe()
    r.config.enable_chunked_prefill = False
    r.config.kv_cache_dtype = "fp8"
    r.generation = 3
    r.parent_id = "abc12345"

    s = r.to_yaml()
    r2 = Recipe.from_yaml(s)
    assert r2.config.enable_chunked_prefill is False
    assert r2.config.kv_cache_dtype == "fp8"
    assert r2.generation == 3
    assert r2.parent_id == "abc12345"


# ---------------------------------------------------------------------------
# Recipe.diff
# ---------------------------------------------------------------------------


def test_recipe_diff_no_changes() -> None:
    r = _make_recipe()
    assert r.diff(r) == {}


def test_recipe_diff_one_change() -> None:
    r1 = _make_recipe()
    r2 = _make_recipe()
    r2.config.enable_prefix_caching = True
    diff = r1.diff(r2)
    assert "enable_prefix_caching" in diff
    assert diff["enable_prefix_caching"]["from"] is False
    assert diff["enable_prefix_caching"]["to"] is True


def test_recipe_diff_multiple_changes() -> None:
    r1 = _make_recipe()
    r2 = _make_recipe()
    r2.config.max_num_seqs = 64
    r2.config.gpu_memory_utilization = 0.8
    diff = r1.diff(r2)
    assert set(diff.keys()) == {"max_num_seqs", "gpu_memory_utilization"}


# ---------------------------------------------------------------------------
# Recipe ID uniqueness
# ---------------------------------------------------------------------------


def test_recipe_ids_are_unique() -> None:
    recipes = [_make_recipe() for _ in range(20)]
    ids = [r.id for r in recipes]
    assert len(set(ids)) == 20

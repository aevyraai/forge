# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for aevyra_forge.cli — unit-testable helpers only.

We do NOT test commands that invoke vLLM or the real orchestrator.
Those require GPU hardware and are covered by Colab integration tests.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from aevyra_forge.cli import _detect_hardware, _make_run_dir


# ---------------------------------------------------------------------------
# _detect_hardware — cpu path (no external tools needed)
# ---------------------------------------------------------------------------


def test_detect_hardware_cpu() -> None:
    from aevyra_forge.recipe import HardwareSpec

    hw = _detect_hardware("cpu")
    assert isinstance(hw, HardwareSpec)
    assert hw.vendor == "cpu"
    assert hw.gpu_type == "cpu"
    assert hw.count == 1
    assert hw.memory_gb_per_gpu == 0


def test_detect_hardware_unknown_device() -> None:
    with pytest.raises(ValueError, match="Unknown device"):
        _detect_hardware("tpu")


# ---------------------------------------------------------------------------
# _detect_hardware — cuda path (mocked nvidia-smi)
# ---------------------------------------------------------------------------


_NVIDIA_SMI_OUTPUT = "Tesla T4, 15360\n"


def test_detect_hardware_cuda_parses_nvidia_smi() -> None:
    from aevyra_forge.recipe import HardwareSpec

    class _FakeResult:
        returncode = 0
        stdout = _NVIDIA_SMI_OUTPUT
        stderr = ""

    with patch("subprocess.run", return_value=_FakeResult()):
        hw = _detect_hardware("cuda")

    assert isinstance(hw, HardwareSpec)
    assert hw.vendor == "nvidia"
    assert hw.gpu_type == "Tesla T4"
    assert hw.count == 1
    assert hw.memory_gb_per_gpu == 15  # round(15360 / 1024)


def test_detect_hardware_cuda_multi_gpu() -> None:
    class _FakeResult:
        returncode = 0
        stdout = "Tesla A100-SXM4-80GB, 81920\nTesla A100-SXM4-80GB, 81920\n"
        stderr = ""

    with patch("subprocess.run", return_value=_FakeResult()):
        hw = _detect_hardware("cuda")

    assert hw.count == 2
    assert hw.memory_gb_per_gpu == 80  # round(81920 / 1024)


def test_detect_hardware_cuda_nvidia_smi_fails() -> None:
    class _FakeResult:
        returncode = 1
        stdout = ""
        stderr = "NVIDIA-SMI has failed"

    with patch("subprocess.run", return_value=_FakeResult()):
        with pytest.raises(RuntimeError, match="Could not detect NVIDIA GPU"):
            _detect_hardware("cuda")


def test_detect_hardware_cuda_no_gpus() -> None:
    class _FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    with patch("subprocess.run", return_value=_FakeResult()):
        with pytest.raises(RuntimeError, match="no GPUs"):
            _detect_hardware("cuda")


# ---------------------------------------------------------------------------
# _detect_hardware — rocm path (mocked rocm-smi)
# ---------------------------------------------------------------------------


_ROCM_SMI_JSON = """{
  "card0": {
    "Card series": "AMD Instinct MI300X",
    "VRAM Total Memory (B)": "206158430208"
  }
}"""


def test_detect_hardware_rocm_parses_rocm_smi() -> None:
    from aevyra_forge.recipe import HardwareSpec

    class _FakeResult:
        returncode = 0
        stdout = _ROCM_SMI_JSON
        stderr = ""

    with patch("subprocess.run", return_value=_FakeResult()):
        hw = _detect_hardware("rocm")

    assert isinstance(hw, HardwareSpec)
    assert hw.vendor == "amd"
    assert "MI300X" in hw.gpu_type
    assert hw.count == 1
    assert hw.memory_gb_per_gpu == 192  # round(206158430208 / 1024^3)


def test_detect_hardware_rocm_fails() -> None:
    class _FakeResult:
        returncode = 1
        stdout = ""
        stderr = "rocm-smi not found"

    with patch("subprocess.run", return_value=_FakeResult()):
        with pytest.raises(RuntimeError, match="Could not detect AMD GPU"):
            _detect_hardware("rocm")


# ---------------------------------------------------------------------------
# _make_run_dir
# ---------------------------------------------------------------------------


def test_make_run_dir_uses_provided(tmp_path: Path) -> None:
    provided = tmp_path / "my_run"
    result = _make_run_dir(provided)
    assert result == provided


def test_make_run_dir_auto_generates(tmp_path: Path) -> None:
    result = _make_run_dir(None)
    # Default is ./runs/<timestamp>
    assert "runs" in str(result)
    assert result.name  # not empty


# ---------------------------------------------------------------------------
# CLI entry point smoke test
# ---------------------------------------------------------------------------


def test_cli_entrypoint_no_args() -> None:
    """aevyra-forge with no args should print help and exit 0."""
    result = subprocess.run(
        [sys.executable, "-m", "aevyra_forge.cli"],
        capture_output=True,
        text=True,
    )
    # typer no_args_is_help=True → exits 0 and prints help
    output = result.stdout + result.stderr
    assert "forge" in output.lower() or "tune" in output.lower() or "usage" in output.lower()


def test_cli_playbook_validate_bundled() -> None:
    """aevyra-forge playbook validate should pass for the bundled playbook."""
    bundled = Path(__file__).parent.parent / "src" / "aevyra_forge" / "playbook.md"
    if not bundled.exists():
        pytest.skip("bundled playbook.md not found")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from aevyra_forge.cli import main; import sys; sys.argv=['forge','playbook','validate']; main()",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "OK" in result.stdout or "sections" in result.stdout

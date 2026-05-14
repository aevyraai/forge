# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for aevyra_forge.playbook."""

from pathlib import Path

import pytest

from aevyra_forge.playbook import Playbook, format_for_agent, load_playbook
from aevyra_forge.recipe import HardwareSpec


# ---------------------------------------------------------------------------
# load_playbook — basic parsing
# ---------------------------------------------------------------------------


SIMPLE_PLAYBOOK = """\
## Search space

Enable prefix caching when prefix_cache_hit_estimate > 0.3.

## Heuristics

Start with max_num_seqs=256.

## Forbidden

Never set gpu_memory_utilization > 0.95.
"""


def test_load_playbook_parses_sections(tmp_path: Path) -> None:
    pb_path = tmp_path / "playbook.md"
    pb_path.write_text(SIMPLE_PLAYBOOK)
    pb = load_playbook(pb_path)
    assert "Search space" in pb.sections
    assert "Heuristics" in pb.sections
    assert "Forbidden" in pb.sections


def test_load_playbook_section_content(tmp_path: Path) -> None:
    pb_path = tmp_path / "playbook.md"
    pb_path.write_text(SIMPLE_PLAYBOOK)
    pb = load_playbook(pb_path)
    assert "prefix caching" in pb.sections["Search space"]
    assert "max_num_seqs=256" in pb.sections["Heuristics"]


def test_load_playbook_stores_source_path(tmp_path: Path) -> None:
    pb_path = tmp_path / "playbook.md"
    pb_path.write_text(SIMPLE_PLAYBOOK)
    pb = load_playbook(pb_path)
    assert pb.source_path == pb_path


def test_load_playbook_raw_markdown(tmp_path: Path) -> None:
    pb_path = tmp_path / "playbook.md"
    pb_path.write_text(SIMPLE_PLAYBOOK)
    pb = load_playbook(pb_path)
    assert pb.raw_markdown == SIMPLE_PLAYBOOK


# ---------------------------------------------------------------------------
# load_playbook — YAML front-matter
# ---------------------------------------------------------------------------


PLAYBOOK_WITH_FRONTMATTER = """\
---
version: 1
layers: [config, quant]
---
## Search space

Content here.
"""


def test_load_playbook_parses_frontmatter(tmp_path: Path) -> None:
    pb_path = tmp_path / "playbook.md"
    pb_path.write_text(PLAYBOOK_WITH_FRONTMATTER)
    pb = load_playbook(pb_path)
    assert pb.metadata.get("version") == 1
    assert pb.metadata.get("layers") == ["config", "quant"]
    assert "Search space" in pb.sections


def test_load_playbook_empty_frontmatter(tmp_path: Path) -> None:
    pb_path = tmp_path / "playbook.md"
    pb_path.write_text("---\n---\n## Heuristics\nContent\n")
    pb = load_playbook(pb_path)
    assert pb.metadata == {}
    assert "Heuristics" in pb.sections


# ---------------------------------------------------------------------------
# load_playbook — empty / edge cases
# ---------------------------------------------------------------------------


def test_load_playbook_empty_file(tmp_path: Path) -> None:
    pb_path = tmp_path / "playbook.md"
    pb_path.write_text("")
    pb = load_playbook(pb_path)
    assert pb.sections == {}


def test_load_playbook_no_sections(tmp_path: Path) -> None:
    pb_path = tmp_path / "playbook.md"
    pb_path.write_text("Just some text without headings.\n")
    pb = load_playbook(pb_path)
    assert pb.sections == {}


def test_load_playbook_empty_constructor() -> None:
    pb = Playbook(raw_markdown="")
    assert pb.sections == {}
    assert pb.metadata == {}


# ---------------------------------------------------------------------------
# load_playbook — bundled default playbook
# ---------------------------------------------------------------------------


def test_bundled_playbook_loads() -> None:
    """The packaged playbook.md must be loadable and have sections."""
    bundled = Path(__file__).parent.parent / "src" / "aevyra_forge" / "playbook.md"
    if not bundled.exists():
        pytest.skip("bundled playbook.md not found")
    pb = load_playbook(bundled)
    assert len(pb.sections) > 0


# ---------------------------------------------------------------------------
# format_for_agent
# ---------------------------------------------------------------------------


MULTI_SECTION_PLAYBOOK = """\
## Search space

Always tune max_num_seqs first.

## Heuristics

T4 needs low gpu_memory_utilization.

## Forbidden

Do not set tensor_parallel_size > GPU count.

## Hardware: nvidia

NVIDIA-specific tuning tips here.

## Hardware: amd

AMD-specific tuning tips here.

## Termination

Stop after 3 convergent experiments.
"""


def test_format_for_agent_includes_always_sections(tmp_path: Path) -> None:
    pb_path = tmp_path / "playbook.md"
    pb_path.write_text(MULTI_SECTION_PLAYBOOK)
    pb = load_playbook(pb_path)
    hw = HardwareSpec(vendor="nvidia", gpu_type="T4", count=1, memory_gb_per_gpu=16)
    text = format_for_agent(pb, hw, layer="config")
    assert "Search space" in text
    assert "Heuristics" in text
    assert "Forbidden" in text
    assert "Termination" in text


def test_format_for_agent_includes_vendor_section(tmp_path: Path) -> None:
    pb_path = tmp_path / "playbook.md"
    pb_path.write_text(MULTI_SECTION_PLAYBOOK)
    pb = load_playbook(pb_path)
    hw = HardwareSpec(vendor="nvidia", gpu_type="T4", count=1, memory_gb_per_gpu=16)
    text = format_for_agent(pb, hw, layer="config")
    assert "NVIDIA-specific" in text
    assert "AMD-specific" not in text


def test_format_for_agent_amd_vendor(tmp_path: Path) -> None:
    pb_path = tmp_path / "playbook.md"
    pb_path.write_text(MULTI_SECTION_PLAYBOOK)
    pb = load_playbook(pb_path)
    hw = HardwareSpec(vendor="amd", gpu_type="MI300X", count=8, memory_gb_per_gpu=192)
    text = format_for_agent(pb, hw, layer="config")
    assert "AMD-specific" in text
    assert "NVIDIA-specific" not in text

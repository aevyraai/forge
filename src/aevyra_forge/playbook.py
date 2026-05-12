# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Playbook loader — the agent's instructions, parsed from markdown.

See AGENT.md → "Key concepts → The playbook".

A playbook is a markdown file with conventional section headings.
``load_playbook`` parses it into a structured ``Playbook`` that the
agent prompt can read selectively (only the sections relevant to the
current hardware + layer).

The playbook is the optimization target for Reflex in v0.2+. For now
it's hand-curated content in ``playbooks/default.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aevyra_forge.recipe import HardwareSpec


@dataclass
class Playbook:
    raw_markdown: str
    sections: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None


def load_playbook(path: str | Path) -> Playbook:
    """Read a markdown playbook from disk and parse it into sections."""
    import yaml as _yaml

    path = Path(path)
    raw = path.read_text()

    # Parse YAML front-matter (--- ... ---)
    metadata: dict[str, Any] = {}
    body = raw
    if raw.startswith("---"):
        end = raw.find("---", 3)
        if end != -1:
            metadata = _yaml.safe_load(raw[3:end]) or {}
            body = raw[end + 3 :].lstrip("\n")

    # Split body into sections on ## headings
    sections: dict[str, str] = {}
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in body.splitlines():
        if line.startswith("## "):
            if current_heading is not None:
                sections[current_heading] = "\n".join(current_lines).strip()
            current_heading = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_heading is not None:
        sections[current_heading] = "\n".join(current_lines).strip()

    return Playbook(raw_markdown=raw, sections=sections, metadata=metadata, source_path=path)


def format_for_agent(playbook: Playbook, hardware: HardwareSpec, layer: str) -> str:
    """Filter the playbook to sections relevant for this hardware + layer."""
    always_include = {"Search space", "Heuristics", "Forbidden", "Termination"}
    hardware_key = f"Hardware: {hardware.vendor}"

    parts: list[str] = []
    for heading, content in playbook.sections.items():
        if heading in always_include or heading.lower() == hardware_key.lower():
            parts.append(f"## {heading}\n\n{content}")

    return "\n\n".join(parts)

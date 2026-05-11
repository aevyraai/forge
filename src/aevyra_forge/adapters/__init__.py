# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Workload-source adapters (Langfuse, OTel, etc.).

v0 thin wrappers; see AGENT.md → "v0 scope" item 4. Real adapters
land alongside the workload module bodies.
"""

from __future__ import annotations

from aevyra_forge.workload import (
    workload_from_jsonl,
    workload_from_langfuse,
    workload_from_otel,
    workload_synthetic,
)


__all__ = [
    "workload_from_jsonl",
    "workload_from_langfuse",
    "workload_from_otel",
    "workload_synthetic",
]

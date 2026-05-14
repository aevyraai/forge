# docs/

User-facing documentation in `.mdx`. Pulled by the aevyra.ai docs site
(Mintlify). Same pattern as `aevyra-origin/docs/`.

## Written

- `quickstart.mdx` — `pip install` → first tuned recipe in 15 min
- `concepts/recipe.mdx` — the artifact (VLLMConfig, layers, lineage)
- `concepts/playbook.mdx` — the agent's instruction manual
- `tutorial-colab-quickstart.mdx` — dry-run and real-run walkthrough with logs

## Planned

- `introduction.mdx` — what Forge is, where it fits in the Aevyra stack
- `tutorial-amd-mi300x.mdx` — AMD-specific walkthrough (blocked: no MI300X access yet)
- `concepts/orchestrator.mdx` — Amdahl scheduling, budget, convergence
- `api/recipe.mdx` — Recipe / VLLMConfig / QuantRecipe reference
- `api/orchestrator.mdx` — Orchestrator / ForgeConfig reference
- `api/bench.mdx` — BenchResult schema

Don't write these until the corresponding code lands — outdated docs
are worse than no docs. Track tutorial-readiness in AGENT.md as the
modules light up.

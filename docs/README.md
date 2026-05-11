# docs/

User-facing documentation in `.mdx`. Pulled by the aevyra.ai docs site
(Mintlify). Same pattern as `aevyra-origin/docs/`.

## Planned

- `introduction.mdx` — what Forge is, where it fits in the Aevyra stack
- `quickstart.mdx` — `pip install` → first tuned recipe in 15 min
- `tutorial-colab-quickstart.mdx` — walk through the Colab notebook
- `tutorial-amd-mi300x.mdx` — AMD-specific walkthrough
- `tutorial-byo-workload.mdx` — bring your own trace
- `concepts/recipe.mdx` — the artifact
- `concepts/playbook.mdx` — the agent's instructions
- `concepts/orchestrator.mdx` — Amdahl scheduling
- `api/recipe.mdx` — Recipe / VLLMConfig / QuantRecipe reference
- `api/orchestrator.mdx` — Orchestrator / ForgeConfig reference
- `api/bench.mdx` — BenchResult schema

Don't write these until the corresponding code lands — outdated docs
are worse than no docs. Track tutorial-readiness in AGENT.md as the
modules light up.

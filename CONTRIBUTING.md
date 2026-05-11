# Contributing to Forge

Forge is built collaboratively, often with help from coding agents
(Sonnet, etc.). The ground rules below keep the architecture coherent
while letting many hands move quickly.

## Before you start

1. **Read [AGENT.md](./AGENT.md) end-to-end.** It's the build spec.
   If you find yourself about to build something that isn't documented
   there, stop and open an issue first.
2. **The public API surface is small and load-bearing.** Adding a new
   public name (anything exported from `aevyra_forge/__init__.py`) is
   a deliberate decision. Document it in AGENT.md before exporting it.
3. **Match conventions from Origin.** Apache 2.0 license header on
   every `.py`; `from __future__ import annotations` at top; type
   hints everywhere; dataclasses for structured values; logging not
   print; private names start with `_`.

## Working with coding agents

If you're using Sonnet (or another LLM) to implement a module:

- Point it at the module's section in AGENT.md as the spec.
- Keep PRs scoped to one module at a time. Cross-module refactors are
  separate PRs.
- The agent's first PR for a new module should fill in the dataclasses
  and signatures, no real logic. Real logic lands in follow-up PRs
  with focused diffs.
- If the agent proposes a public API change, that's a flag — get human
  review before merging.

## What's tracked where

- **`AGENT.md`** — architecture, contracts, conventions, what's in
  scope vs. out of scope. The single source of truth.
- **`playbooks/*.md`** — agent instructions. These are *content*, not
  code. Review them like prompt-engineering changes: small commits,
  before/after benchmark numbers, narrative rationale.
- **`notebooks/*.ipynb`** — Colab/Kaggle-runnable demos. These ship
  with the package and must always work end-to-end against the public
  release on PyPI. CI runs them on every PR.
- **`docs/*.mdx`** — user-facing documentation. The Mintlify site
  pulls from here.

## Vendor-neutrality

This is a hard rule. **Any new code path that only works on one
vendor needs a parallel implementation or an explicit fallback in
the same PR.** Don't merge NVIDIA-only paths without an AMD plan;
don't merge AMD-only paths without a NVIDIA plan. If both can't be
done at once, gate the new path behind a check and document the
imbalance.

## Tests

Tests will land alongside real module bodies. Until then, lint and
type-check (`ruff check`, `mypy --strict`) must always pass. CI is
green-only — never merge red.

## Layer scope

Forge is layered: Layer 1 (config), Layer 2 (quant), Layer 3 (kernel
synthesis). v0 is Layer 1 only.

- **Layer 2 contributions are welcome** but must come with a
  benchmark showing they preserve accuracy. Quantization that tanks
  accuracy is worse than no quantization.
- **Layer 3 is an AutoKernel integration**, not a from-scratch
  build. PRs that re-implement kernel synthesis inside Forge will
  be redirected.

## License

Apache 2.0. Every `.py` file gets the standard header (copy from
`recipe.py`).

## Code review

PRs need one approval from a maintainer. Coding-agent-authored PRs
need a human review of the *architecture* (does it match AGENT.md?)
even if the code itself is reviewed by another agent. The architecture
is the load-bearing thing; the code is replaceable.

## Releases

`0.0.x` — pre-alpha, breaking changes any time.
`0.1.x` — v0 complete: Layer 1 working end-to-end with one playbook.
`0.2.x` — Layer 2.
`0.3.x` — Layer 3 + engine adapters beyond vLLM.

We tag releases manually; no automation yet.

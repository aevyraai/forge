# Examples

Runnable Python scripts for each major Forge use case. Mirrors the
pattern used in `aevyra-origin/examples/`.

## Planned

- **`quickstart/`** — minimal `forge tune` invocation against a
  synthetic workload. The fastest path to seeing the loop run.

- **`chat_workload/`** — chat-shaped workload (short prompts, ~1k
  output tokens, high prefix-cache hit rate). Demonstrates how
  `enable_prefix_caching` becomes the dominant win.

- **`long_context/`** — long-context workload (40k+ input tokens,
  short outputs). Demonstrates `block_size=32` and chunked prefill
  trade-offs.

- **`production_trace/`** — bring-your-own-workload example using
  a captured Langfuse export.

Each example has its own `README.md`, `pipeline_workload.jsonl` (or a
synthetic-generation script), and a `recipe_baseline.yaml` to diff
against.

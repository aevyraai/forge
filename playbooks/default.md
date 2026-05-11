---
name: default
version: 0.0.1
layer: config
status: stub
---

# Forge default playbook (Layer 1 only)

> **Status:** stub. Sonnet (or a human contributor) fills this in
> alongside the Layer 1 implementation. The shape is right; the
> content is not yet load-bearing.

This playbook is the agent's instructions. It is parsed by
`aevyra_forge.playbook.load_playbook` and selectively rendered into
the agent's prompt by `format_for_agent`. Every section heading is a
hard contract — the parser looks for these exactly.

---

## Search space

The Layer 1 knobs and their legal ranges. The agent must propose
mutations whose values fall in these ranges.

| Knob | Range | Notes |
|---|---|---|
| `max_num_seqs` | 16, 32, 64, 128, 256, 512, 1024 | Higher = more throughput, more KV pressure |
| `max_num_batched_tokens` | 2048, 4096, 8192, 16384 | Must be ≥ `max_num_seqs` × `block_size` typically |
| `block_size` | 16, 32 | 32 wins on H100 for long contexts |
| `gpu_memory_utilization` | 0.85 - 0.95 | Above 0.95 = OOM risk |
| `enable_prefix_caching` | true, false | Big win for chat workloads with shared system prompt |
| `enable_chunked_prefill` | true, false | Default on; off only for very latency-sensitive workloads |
| `kv_cache_dtype` | auto, fp8, bf16 | fp8 needs Hopper+ |
| `swap_space` | 0, 4, 8, 16 (GiB) | Trade GPU memory for swap to disk |
| `tensor_parallel_size` | depends on `hardware.count` | TP must divide the GPU count |

---

## Heuristics

If-then rules. Tagged so the agent can quote the rule it followed.

- **H1**: If `max_num_seqs` × p99 input tokens > GPU memory budget,
  reducing `max_num_seqs` will reduce OOM risk more than reducing
  `gpu_memory_utilization`.
- **H2**: If the workload's prefix-cache hit estimate is > 30%,
  `enable_prefix_caching=true` is almost always a win.
- **H3**: Chunked prefill helps when input tokens vary widely. Turn
  off only for tightly-shaped workloads where the overhead matters.
- **H4**: Start with the vLLM defaults. Don't change more than two
  knobs in a single experiment — attribution becomes ambiguous.

---

## Hardware: nvidia

NVIDIA-specific guidance.

- `attention_backend`: prefer `flash-attn-3` on Hopper+; default
  elsewhere.
- FP8 KV cache requires Hopper or later.
- T4 (Colab free tier): cap `max_num_seqs` at 64, `block_size=16`.
- A100: `kv_cache_dtype=auto` is usually right; consider FP8 only
  if accuracy validation passes.
- H100/B200: FP8 KV cache typically gives 20-30% throughput with
  negligible accuracy delta on chat workloads.

## Hardware: amd

AMD-specific guidance.

- `attention_backend`: prefer `ROCM_AITER_FA` or `AITER_MLA` on
  MI300X / MI325X — usually 1.2-4× faster than the default backend.
- FP8 on MI300X is OCP FP8 (not E4M3); calibration data differs.
- `block_size=32` often wins on MI300X for long-context workloads.
- vLLM-ROCm has 7 attention backends as of vLLM 0.x — when in doubt,
  try the AITER variants first.

## Hardware: intel

Stub. Gaudi 3 support is planned.

## Hardware: google

Stub. Trainium/TPU support is post-v0.

---

## Forbidden

Combinations the agent must avoid. Quoting the constraint by name
in the rationale is encouraged.

- **F1**: `speculative_model` with `tensor_parallel_size > 4`
  (vLLM 0.x instability — revisit when fixed upstream).
- **F2**: `gpu_memory_utilization > 0.97` (OOM rate spikes).
- **F3**: `block_size=8` on workloads with input tokens > 4k
  (severe fragmentation).

---

## Termination

When to stop. The orchestrator checks these between experiments.

- **T1**: 5 consecutive experiments with score improvement < 1% →
  declare convergence on this layer.
- **T2**: Total wall-clock > `max_wall_clock_hours` → stop.
- **T3**: Total experiments > `max_experiments` → stop.
- **T4**: 3 consecutive CRASH experiments → stop and surface
  diagnostics; something is broken at the runner level, not at the
  recipe level.

---
name: default
version: 0.1.0
layer: config
status: active
---

# Forge default playbook (Layer 1 — vLLM config)

This playbook is the agent's instructions. It is parsed by
`aevyra_forge.playbook.load_playbook` and selectively rendered into
the agent's prompt by `format_for_agent`.

---

## Search space

Layer 1 knobs and their legal ranges. Propose values ONLY from these sets.

| Knob | Legal values | Notes |
|---|---|---|
| `max_num_seqs` | 8, 16, 32, 64, 128, 256, 512 | See hardware section for per-GPU caps |
| `max_num_batched_tokens` | 2048, 4096, 8192, 16384, 32768 | Must be ≥ max_num_seqs × avg_input_tokens |
| `block_size` | 16, 32 | 32 wins on H100/MI300X; 16 is safer on low-VRAM GPUs |
| `gpu_memory_utilization` | 0.80, 0.85, 0.88, 0.90, 0.92, 0.95 | See F2 — do not exceed 0.95 |
| `enable_prefix_caching` | true, false | See H2 |
| `enable_chunked_prefill` | true, false | Default true; see H3 |
| `kv_cache_dtype` | auto, fp8 | fp8 requires Hopper (H100/H200) or MI300X |
| `tensor_parallel_size` | 1, 2, 4, 8 | Must divide hardware.count evenly |

---

## Memory budget (REQUIRED reading before proposing any knob change)

Before proposing a mutation that touches `max_num_seqs`,
`max_num_batched_tokens`, `gpu_memory_utilization`, or `block_size`,
reason through this formula:

```
kv_cache_budget_gb = gpu_memory_utilization × vram_per_gpu_gb − model_weight_gb
max_safe_seqs ≈ kv_cache_budget_gb × 1024 / (2 × n_layers × n_heads × head_dim × block_size / 1024)
```

Approximate model weight footprints (fp16):
- 1–3B params  → ~3 GB
- 7–8B params  → ~15 GB
- 13B params   → ~26 GB
- 34B params   → ~68 GB
- 70B params   → ~140 GB

Rules derived from the formula:
- **Raising max_num_seqs requires MORE kv_cache_budget** — it DOES NOT help
  to simultaneously lower gpu_memory_utilization.
- **If an OOM crash occurred**: the current (max_num_seqs, gpu_memory_utilization)
  pair exceeds budget. Fix by REDUCING max_num_seqs OR INCREASING
  gpu_memory_utilization — never both in the wrong direction.
- **Never pair max_num_seqs > current_value with gpu_memory_utilization < current_value.**

---

## Heuristics

If-then rules. Quote the rule ID in your rationale.

- **H1**: OOM crash → reduce `max_num_seqs` by one step (e.g. 64→32) OR
  increase `gpu_memory_utilization` by 0.02–0.05. Do not change both at once.
- **H2**: Workload prefix-cache hit estimate > 30% (or shared system prompt) →
  `enable_prefix_caching=true` is almost always a 15-40% throughput win.
- **H3**: Chunked prefill (`enable_chunked_prefill=true`) helps when input
  length varies widely (p99/p50 > 2×). Turn off only for tightly-shaped
  workloads where batch formation overhead matters.
- **H4**: Change at most ONE knob per experiment unless you have a specific
  reason to change two. Attribution is impossible across 3+ simultaneous changes.
- **H5**: `max_num_batched_tokens` below `max_num_seqs × avg_input_tokens`
  creates artificial batching bottlenecks — raise it first before raising seqs.
- **H6**: If the last 3 experiments were all REVERTED with similar scores,
  the config layer has converged — consider returning empty changes.

---

## Hardware: nvidia

### T4 (16 GB VRAM — Colab free tier / budget cloud)

Memory is tight. Be conservative.

- Model weight budget for 8B models: ~15 GB fp16, leaving only ~1 GB for KV cache
  at `gpu_memory_utilization=0.90`. This means **max_num_seqs ≤ 32** for 8B models.
- `max_num_seqs=64` will OOM with any 7B+ model on T4 — do not propose it.
- `block_size=16` is mandatory (32 doubles KV memory per block).
- `gpu_memory_utilization=0.90` is the safe ceiling on T4; 0.92+ will OOM randomly.
- `enable_prefix_caching=true` is often a win for shared-prompt workloads even on T4.
- Recommended starting point: `max_num_seqs=16, max_num_batched_tokens=4096`.

### A10G (24 GB VRAM — AWS g5, SageMaker)

- 8B models: `max_num_seqs` up to 64 is safe at `gpu_memory_utilization=0.90`.
- 13B models: `max_num_seqs` ≤ 32 at 0.90.
- `block_size=16` or `block_size=32` both work.
- FP8 KV cache NOT supported (Ampere, not Hopper).

### A100 (40/80 GB VRAM)

- 80 GB: 8B models leave ~65 GB for KV cache; `max_num_seqs=256` is reasonable.
- 40 GB: treat like a large A10G; cap `max_num_seqs=128` for 8B models.
- `kv_cache_dtype=auto` is right; consider `fp8` only if accuracy passes.
- `block_size=16` or `32` — benchmark both.

### H100 / H200 (80 GB VRAM — Hopper)

- `attention_backend=flash-attn-3` gives 10-20% win on H100.
- `kv_cache_dtype=fp8` typically gives 20-30% throughput, negligible accuracy delta.
- `max_num_seqs` up to 512 for 8B models; 128 for 70B.
- `block_size=32` wins for long-context workloads.

### V100 (16/32 GB VRAM)

- No FP8, no flash-attn-3.
- Treat 16 GB V100 like T4 for memory budgets.
- `block_size=16` mandatory.

---

## Hardware: amd

### MI300X (192 GB HBM3)

- Massive memory headroom: `max_num_seqs=512` is safe for most models.
- `attention_backend`: prefer `ROCM_AITER_FA` or `AITER_MLA` — usually 1.2–4×
  faster than default. Try `ROCM_AITER_FA` first.
- `block_size=32` often wins for long-context workloads.
- `kv_cache_dtype=fp8` uses OCP FP8 (not E4M3) — calibration data differs from NVIDIA.
- vLLM-ROCm has 7 attention backends — when in doubt, try AITER variants first.

### MI250X (128 GB HBM2e)

- Similar memory profile to MI300X but ~30% lower bandwidth.
- `attention_backend=ROCM_AITER_FA` still recommended.

---

## Hardware: intel

Stub. Gaudi 3 support is planned.

---

## Hardware: google

Stub. Trainium/TPU support is post-v0.

---

## Forbidden

Combinations the agent must never propose. Quote the constraint ID in rationale.

- **F1**: `speculative_model` with `tensor_parallel_size > 4` (vLLM 0.x instability).
- **F2**: `gpu_memory_utilization > 0.95` — OOM rate spikes sharply above this.
- **F3**: `block_size=8` on workloads with input tokens > 4k (fragmentation).
- **F4**: `max_num_seqs=64` or higher on T4 with any 7B+ model (guaranteed OOM).
- **F5**: `kv_cache_dtype=fp8` on Volta, Turing, or Ampere GPUs (hardware unsupported).
- **F6**: `max_num_seqs > max_num_batched_tokens` (vLLM will reject this).

---

## Termination

When to stop. The orchestrator checks these; you may also return empty changes early.

- **T1**: 5 consecutive experiments with score improvement < 1% → converged on this layer.
- **T2**: Total wall-clock > `max_wall_clock_hours` → stop.
- **T3**: Total experiments > `max_experiments` → stop.
- **T4**: 3 consecutive CRASH experiments → stop; something is broken at the runner level.
- **T5**: All promising single-knob mutations have been tried and reverted → return empty changes.

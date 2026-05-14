# AGENT.md

> Build spec for Forge. Read this end-to-end before writing any code.
> Sonnet (or any contributor) implements against this document. Module
> file paths and public function signatures are the contracts.

---

## Project overview

`aevyra-forge` is the **deployment-tuning** layer of the Aevyra stack —
an autonomous overnight optimizer for LLM inference deployments. Give
Forge a model, a hardware target, and a workload trace; it returns the
deployment recipe (vLLM config + quantization scheme + kernel set) that
maximizes throughput at your accuracy and latency constraints.

```
Witness  →  captures what happened           (aevyra-witness)
Verdict  →  judges it                        (aevyra-verdict)
Origin   →  finds where it went wrong        (aevyra-origin)
Reflex   →  fixes the prompts                (aevyra-reflex)
Forge    →  tunes the deployment             (this package)
```

Forge is the **commercial wedge** of the Aevyra stack. It produces an
artifact (the tuned deployment recipe) that's directly valuable to any
team self-hosting LLMs — and it generates the trace volume that feeds
back into Origin attribution and Reflex playbook optimization.

### The five-ingredient pattern

Forge follows the AutoKernel-style autoresearch recipe:

1. **Single artifact** — `recipe.yaml` — vLLM config + quant choice +
   kernel selection. The agent edits this one file.
2. **Fast verifier** — `bench.py` — replays the workload against a
   running vLLM server; returns structured perf + accuracy.
3. **Comprehensive playbook** — `playbook.md` — encodes serving
   expertise. The agent reads this to decide what to try next.
4. **Tight keep/revert loop** — `~5-15 min` per experiment;
   30-100 experiments overnight on a single GPU.
5. **Orchestrator** — Amdahl-style scheduler. Spends experiments where
   they have the most leverage on the current bottleneck.

### The three layers

Forge tunes across three interacting axes in Amdahl order (cheap first,
expensive last):

| Layer | What it tunes | Experiment cost | Typical gain |
|---|---|---|---|
| **1. Config** | vLLM serving args (`max_num_seqs`, KV cache fraction, chunked prefill, prefix caching, spec decode, TP/PP size) | ~5 min | 30-50% |
| **2. Quantization** | INT4/FP8/INT8, KV cache precision, per-layer hints, calibration data | ~15-30 min | Additional 20-40% |
| **3. Kernel synthesis** | Custom kernels for the model's hot paths on the target hardware | ~30-90 min | Additional 10-30% (ceiling uncapper) |

**v0 ships Layer 1 only.** Layer 2 and Layer 3 are scaffolded for
future implementation; their module files exist but raise
`NotImplementedError`.

### The differentiators (why Forge vs. FireOptimizer / Baseten / NIM)

- **Autonomous loop**, not a managed service. Customers run Forge on
  their own hardware, not on ours.
- **On-prem / air-gapped**, not platform-locked. Runs anywhere vLLM
  runs.
- **Vendor-neutral**. AMD MI300X is a first-class target, not a
  port. NVIDIA still works.
- **Workload-aware**. Optimizes against captured production traces,
  not ShareGPT.
- **Self-improving playbook**. The `playbook.md` is itself a Reflex
  optimization target — across many overnight runs, the playbook
  learns which heuristics actually work for which model/hardware
  combos. (This closes the loop with the rest of the Aevyra stack.)

---

## Implementation status

| Module | Status | Notes |
|---|---|---|
| `recipe.py` | ✅ Done | Full YAML/dict round-trip, diff, lineage |
| `workload.py` | ✅ Done | JSONL, synthetic, shared-prefix, concurrent generators; concurrency param; path metadata |
| `playbook.py` | ✅ Done | Section parser, YAML front-matter, `format_for_agent` |
| `result.py` | ✅ Done | `ForgeStore` + `ForgeRun`; run persistence, resume, TSV/JSON rendering |
| `orchestrator.py` | ✅ Done | Main loop, `ForgeConfig`, `Experiment`; persists llm_provider/device/workload_path |
| `cli.py` | ✅ Done | `aevyra-forge tune/resume/report/playbook`; `--device cuda\|rocm\|cpu`; GPU auto-detect |
| `runner.py` | ✅ Done | vLLM subprocess lifecycle; 600s startup timeout |
| `bench.py` | ✅ Done | Async concurrent replay; streaming TTFT measurement |
| `agent.py` | ✅ Done | Single LLM call, structured JSON output |
| `llm.py` | ✅ Done | `provider/model` factory; Anthropic + OpenAI-compat |
| `config.py` | 🔧 Partial | Search space defined; mutation wired |
| `quant.py` | 🚧 Stub | `NotImplementedError` — v0.2 |
| `kernel.py` | 🚧 Stub | `NotImplementedError` — v0.3 |
| `tests/` | ✅ Done | 79 unit tests; no GPU required |

---

## Architecture

```
src/aevyra_forge/
├── __init__.py            # Public API exports
├── recipe.py              # Recipe dataclass + YAML (de)serialization
├── playbook.py            # Playbook loader (parse .md, expose to agent)
├── orchestrator.py        # The search loop. Amdahl scheduler.
├── agent.py               # The LLM-driven decision agent
├── bench.py               # vLLM benchmark wrapper. Workload replay, perf + accuracy.
├── workload.py            # Workload trace ingestion (JSONL, synthetic, Langfuse adapter)
├── runner.py              # vLLM server lifecycle: start, warmup, stop, restart-with-new-recipe
├── config.py              # Layer 1: vLLM config search space + mutation operators
├── quant.py               # Layer 2: quantization recipe selection    [v0: stub]
├── kernel.py              # Layer 3: AutoKernel integration hook      [v0: stub]
├── result.py              # ForgeStore + ForgeRun: run persistence, resume, audit trail
├── llm.py                 # LLMFn type + provider/model factory (Anthropic, OpenAI-compat, local)
├── playbook.md            # Bundled default playbook (copied from playbooks/default.md)
└── cli.py                 # CLI entrypoint (typer): aevyra-forge tune/resume/report/playbook
```

Public API surface (from `aevyra_forge`):

```python
Recipe,           # the artifact
Experiment,       # one keep/revert iteration
Playbook,         # parsed playbook
Orchestrator,     # the scheduler
ForgeConfig,      # budget + termination knobs
BenchResult,      # structured perf+accuracy output
Workload,         # workload trace
ForgeStore,       # multi-run directory manager
ForgeRun,         # handle for one run's on-disk directory
```

From `aevyra_forge.workload`:

```python
workload_from_jsonl               # JSONL → Workload (with concurrency param + path metadata)
workload_from_langfuse            # Langfuse export → Workload (thin wrapper)
workload_synthetic                # ShareGPT-shaped fallback
workload_shared_prefix            # Prefix-cache benchmark workload
workload_concurrent_synthetic     # High-concurrency stress workload
```

---

## Key concepts

### The recipe — single artifact, multi-layer

`Recipe` is a dataclass that the orchestrator mutates and the runner
applies. Conceptually:

```python
@dataclass
class Recipe:
    model: str                              # e.g. "meta-llama/Llama-3.1-8B-Instruct"
    hardware: HardwareSpec                  # GPU vendor + type + count + memory

    # Layer 1: vLLM serving config
    config: VLLMConfig                      # max_num_seqs, KV cache fraction, etc.

    # Layer 2: quantization (v0: defaults only)
    quant: QuantRecipe = field(default_factory=QuantRecipe)

    # Layer 3: custom kernels (v0: empty)
    kernels: list[KernelOverride] = field(default_factory=list)

    # Provenance
    parent_id: str | None = None            # which recipe this was mutated from
    generation: int = 0                     # how many keep/revert cycles deep
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
```

YAML-serializable. The user reads it. The user copies it into their own
deployment. That's the deliverable.

### The playbook — heuristics + search space, not LLM weights

`playbook.md` is the agent's instructions. It contains:

- Description of the search space at each layer
- Heuristics: "if prefix_cache_hit_estimate > 0.3, try enable_prefix_caching first"
- Hardware-specific guidance: "on MI300X, ROCM_AITER_FA is usually faster
  than the default flash-attn backend"
- Forbidden combinations: "speculative decoding with TP > 4 is unstable
  on vLLM 0.x.y"
- Termination conditions: "stop when 5 consecutive experiments don't
  improve score by 1%"

The playbook ships bundled at `src/aevyra_forge/playbook.md` (copied
from `playbooks/default.md`) so it works after `pip install` with no
file path required.

**The playbook is itself an optimization target for Reflex in v0.2+** —
but for v0, it's hand-curated.

### The agent — reads trace, picks next experiment

The agent is a single LLM call per experiment:

```python
@dataclass
class AgentDecision:
    rationale: str
    mutation: dict[str, Any]
    expected_throughput_delta_pct: float | None
    expected_accuracy_delta: float | None
    raw_response: str
```

It produces a JSON-shaped mutation:

```json
{
  "rationale": "Workload shows prefix_cache_hit_estimate=0.82. Enabling prefix caching should reduce TTFT significantly on repeated system prompts.",
  "mutation": {
    "layer": "config",
    "changes": {"enable_prefix_caching": true}
  },
  "expected_throughput_delta_pct": 25,
  "expected_accuracy_delta": 0.0
}
```

The agent's rationale and prediction are logged per experiment. After
bench, the actual outcome is stored alongside the prediction — a signal
Reflex can use later to improve the playbook.

### The orchestrator — Amdahl, not Bayesian

Standard hyperparameter-optimization tools (Optuna, Ray Tune) use
Bayesian methods or evolutionary search. Forge does not. Instead:

1. The agent (LLM) reads the experiment history and proposes the next
   experiment.
2. The orchestrator's job is **scheduling**: which layer to spend the
   next experiment on, when to escalate from Layer 1 to Layer 2, when
   to stop.
3. Scheduling uses Amdahl reasoning: if Layer 1 hasn't moved the score
   in N experiments, escalate to Layer 2.

The orchestrator owns budget enforcement, convergence detection,
layer-escalation decisions, and statistical significance gates.

### The verifier — vLLM benchmark, workload replay

`bench.py` replays the workload against the running vLLM server using
an async `httpx` client at full concurrency. Returns:

```python
@dataclass
class BenchResult:
    throughput_tokens_per_sec: float
    p50_latency_ms: float
    p99_latency_ms: float
    ttft_ms: float                          # time to first token (streaming)
    tpot_ms: float                          # time per output token
    peak_vram_mb: int
    max_concurrent_seqs: int
    accuracy_score: float | None
    accuracy_delta_vs_baseline: float | None
    recipe_id: str
    workload_id: str
    bench_duration_s: float
    status: Literal["PASS", "FAIL", "TIMEOUT", "CRASH"]
    error: str | None = None
```

The score function for v0: `throughput_tokens_per_sec` subject to
`accuracy_score >= accuracy_floor`. Experiments that regress accuracy
below the floor are not kept regardless of throughput gains.

### The runner — vLLM server lifecycle

Each experiment:
1. Boot vLLM subprocess with the recipe's config applied.
2. Wait up to 600 seconds for `/health` (first run includes weight
   download).
3. Run bench at `workload.concurrency` simultaneous in-flight requests.
4. Stop the server (`SIGTERM` + drain).

Startup timeout is 600s by default (`ForgeConfig.vllm_startup_timeout_s`)
to handle first-run weight downloads on Colab.

### The workload — production trace or synthetic fallback

A `Workload` is a stream of `(prompt, expected_output_tokens)` pairs
with arrival timing and a `concurrency` cap. Sources, in order of
preference:

1. **Production trace** — customer's actual traffic via JSONL.
   `workload_from_jsonl` stores the resolved file path in
   `metadata["path"]` so resume can find it without re-specifying
   `--workload`.
2. **Synthetic shape match** — `workload_shared_prefix`,
   `workload_concurrent_synthetic` for specific benchmark patterns.
3. **ShareGPT / public benchmark** — `workload_synthetic` fallback for
   the Colab demo.

`concurrency` (default 8) is persisted to `config.json` at run start
and restored on `aevyra-forge tune resume`. T4/A10: 8–16.
A100/H100: 32–64.

### Run persistence — ForgeStore and ForgeRun

```
.forge/
  runs/
    001_2026-05-13T04-10-00/
      config.json          ← model, hardware, workload_path, concurrency, llm_provider, all ForgeConfig fields
      experiments.jsonl    ← append-only log (one line per experiment)
      experiments.tsv      ← human-readable table (re-written each append)
      experiments.json     ← structured table for tooling (re-written each append)
      best_recipe.yaml     ← best config so far (updated when score improves)
      completed.json       ← written on clean finish; absent = interrupted
    002_2026-05-13T14-05-00/
      ...
```

`ForgeStore` manages the directory of runs. `ForgeRun` is a handle for
one run's directory. A run with `experiments.jsonl` but no
`completed.json` was interrupted and can be resumed.

`aevyra-forge tune resume` calls `ForgeStore.find_incomplete_run()` and
reads all parameters from `config.json` — **no CLI args needed**.

### Hardware auto-detection

`_detect_hardware(device)` in `cli.py` queries `nvidia-smi` (cuda) or
`rocm-smi` (rocm) at runtime to auto-detect GPU name, VRAM, and count.
No lookup table. Works for any GPU that the smi tool knows about.

```python
# cuda
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
# → "Tesla T4, 15360"  (name, MiB)

# rocm
rocm-smi --showproductname --showmeminfo vram --json
# → {"card0": {"Card series": "AMD Instinct MI300X", "VRAM Total Memory (B)": "..."}}
```

`--device cpu` returns a placeholder `HardwareSpec` for dry-run use.

### Vendor neutrality — AMD MI300X is a first-class target

The recipe schema includes per-vendor fields (`attention_backend` differs:
`flash-attn-3` on Hopper, `ROCM_AITER_FA` on MI300X). The playbook has
`## Hardware: nvidia` and `## Hardware: amd` sections filtered per run.
The `quant` module will handle FP8 differently for E4M3 (Hopper) vs.
OCP FP8 (MI300X+) in v0.2.

---

## Module-by-module spec

### `recipe.py`

```python
@dataclass
class HardwareSpec:
    vendor: Literal["nvidia", "amd", "intel", "google", "other", "cpu"]
    gpu_type: str                            # "T4", "A100", "MI300X", ...
    count: int
    memory_gb_per_gpu: int

    def label(self) -> str:                  # "nvidia/T4x1"

@dataclass
class VLLMConfig:
    """Layer 1 search space. All fields tunable; defaults are vLLM's defaults."""
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    block_size: int = 16
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = True
    swap_space: int = 4
    kv_cache_dtype: Literal["auto", "fp8", "fp16", "bf16"] = "auto"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    speculative_model: str | None = None
    num_speculative_tokens: int = 0
    attention_backend: str | None = None     # vendor-specific

@dataclass
class QuantRecipe:
    """Layer 2. v0: defaults only."""
    method: Literal["fp16", "bf16", "int4_awq", "int4_gptq", "fp8_e4m3", "int8"] = "bf16"
    kv_cache_quant: Literal["none", "fp8", "int8"] = "none"
    calibration_dataset: str | None = None
    per_layer_overrides: dict[str, str] = field(default_factory=dict)

@dataclass
class KernelOverride:
    """Layer 3. v0: never populated."""
    op_name: str
    kernel_source_path: str

@dataclass
class Recipe:
    model: str
    hardware: HardwareSpec
    config: VLLMConfig = field(default_factory=VLLMConfig)
    quant: QuantRecipe = field(default_factory=QuantRecipe)
    kernels: list[KernelOverride] = field(default_factory=list)
    parent_id: str | None = None
    generation: int = 0
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict[str, Any]: ...
    def to_yaml(self) -> str: ...
    @classmethod
    def from_dict(cls, d: dict) -> "Recipe": ...
    @classmethod
    def from_yaml(cls, s: str) -> "Recipe": ...
    def diff(self, other: "Recipe") -> dict[str, Any]:
        """Return VLLMConfig fields that differ between self and other."""
```

### `workload.py`

```python
@dataclass
class WorkloadRequest:
    prompt: str
    expected_output_tokens: int
    arrival_offset_s: float
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class Workload:
    requests: list[WorkloadRequest]
    duration_s: float
    metadata: dict[str, Any] = field(default_factory=dict)
    concurrency: int = 1                   # max simultaneous in-flight during bench

    def min_context_tokens(self) -> int:
        """Minimum max_model_len needed to serve this workload."""
    def summary(self) -> dict[str, Any]:
        """Distributions the agent prompt reads (p50/p99 lengths, RPS, prefix hit estimate)."""

def workload_from_jsonl(path: str | Path, *, concurrency: int = 8) -> Workload:
    """Load from JSONL. Stores resolved path in metadata["path"] for resume."""

def workload_from_langfuse(path: str | Path) -> Workload:
    """Thin wrapper around workload_from_jsonl."""

def workload_from_otel(spans: list[dict]) -> Workload:
    """v0: stub — raises NotImplementedError."""

def workload_synthetic(*, n_requests=1000, avg_input_tokens=512, avg_output_tokens=128,
                       duration_s=60.0, seed=0) -> Workload: ...

def workload_shared_prefix(*, n_requests=200, concurrency=8, prefix_tokens=512,
                           query_tokens=64, output_tokens=128, duration_s=60.0, seed=0) -> Workload:
    """All requests share a long common prefix — benchmarks prefix caching."""

def workload_concurrent_synthetic(*, n_requests=200, concurrency=16, avg_input_tokens=256,
                                  avg_output_tokens=128, duration_s=60.0, seed=0) -> Workload:
    """High-concurrency workload — stresses max_num_seqs and batching knobs."""
```

### `result.py`

```python
class ForgeRun:
    """Handle for one optimization run's on-disk directory."""

    path: Path
    run_id: str

    def save_config(self, *, model, hardware_label, workload_id, forge_config_dict,
                    llm_provider="", device="cuda", workload_path="", concurrency=8) -> None:
        """Write config.json once at run start."""

    def config(self) -> dict[str, Any] | None: ...
    def append(self, exp: Experiment) -> None:
        """Atomic JSONL append + re-render TSV, JSON table, best_recipe.yaml."""
    def save_completion(self, *, best_score, n_kept, n_total, wall_time_s) -> None:
        """Write completed.json — marks the run as cleanly finished."""
    def history(self) -> list[Experiment]: ...
    def best(self) -> Experiment | None: ...
    def is_complete(self) -> bool: ...
    def is_interrupted(self) -> bool: ...
    def status(self) -> str: ...        # "running" | "interrupted" | "completed"
    def render_tsv(self) -> str: ...
    def summary_row(self) -> dict[str, Any]: ...

# Backward-compat alias — old code that imports ExperimentStore still works
ExperimentStore = ForgeRun

class ForgeStore:
    """Manages a directory of Forge optimization runs."""

    def __init__(self, root: str | Path = ".forge"): ...
    def new_run(self) -> ForgeRun: ...
    def get_run(self, run_id: str) -> ForgeRun | None: ...
    def latest_run(self) -> ForgeRun | None: ...
    def find_incomplete_run(self) -> ForgeRun | None:
        """Return the most recent interrupted run — used by `aevyra-forge tune resume`."""
    def list_runs(self) -> list[dict[str, Any]]: ...
```

### `orchestrator.py`

```python
@dataclass
class Experiment:
    id: str
    recipe: Recipe
    bench_result: BenchResult | None = None
    agent_decision: AgentDecision | None = None
    score: float | None = None
    kept: bool = False
    duration_s: float = 0.0
    started_at: float = 0.0
    ended_at: float = 0.0
    llm_tokens: int = 0

@dataclass
class ForgeConfig:
    max_experiments: int = 50
    max_wall_clock_hours: float = 12.0
    max_dollars: float | None = None
    accuracy_floor: float = 0.99
    min_improvement_pct: float = 1.0
    convergence_window: int = 5
    layer_escalation: bool = True
    exploration_interval: int = 4
    dry_run: bool = False
    vllm_startup_timeout_s: int = 600      # 10 min — first start includes weight download

class Orchestrator:
    def __init__(
        self,
        *,
        model: str,
        hardware: HardwareSpec,
        workload: Workload,
        playbook: Playbook,
        llm: LLMFn,
        store: ForgeStore,
        forge_config: ForgeConfig = ForgeConfig(),
        llm_provider: str = "",            # persisted to config.json for resume
        device: str = "cuda",              # persisted to config.json for resume
        workload_path: str = "",           # persisted to config.json for resume
    ): ...

    def run(self) -> tuple[Recipe, list[Experiment]]:
        """Start a new run. Returns best recipe + full history."""

    def resume(self) -> tuple[Recipe, list[Experiment]]:
        """Continue an interrupted run from its ForgeStore state."""
```

### `cli.py`

```bash
# Start a new run
aevyra-forge tune \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --device cuda \
  --workload examples/sample_workload.jsonl \
  --concurrency 8 \
  --llm anthropic/claude-sonnet-4-6 \
  --max-experiments 50 \
  --max-hours 12.0

# Resume latest interrupted run (zero args — all config read from config.json)
aevyra-forge tune resume

# View results
aevyra-forge report .forge/

# Inspect the bundled playbook
aevyra-forge playbook show
aevyra-forge playbook validate
```

`--device cuda` triggers `nvidia-smi` auto-detection of GPU name, VRAM,
and count. `--device rocm` uses `rocm-smi`. `--device cpu` returns a
placeholder (use with `--dry-run`).

All CLI flags except `--model`, `--workload`, and `--device` have
defaults and are persisted to `config.json` at run start so
`aevyra-forge tune resume` needs zero arguments.

### `agent.py`

```python
@dataclass
class AgentDecision:
    rationale: str
    mutation: dict[str, Any]
    expected_throughput_delta_pct: float | None
    expected_accuracy_delta: float | None
    raw_response: str
```

The prompt template (`_AGENT_PROMPT`) includes: workload summary,
hardware spec, memory budget, current recipe YAML, last N experiment
diffs + scores, search space, and the full playbook. Output is strict
JSON. The agent makes exactly one LLM call per experiment.

### `bench.py`

```python
async def _bench_async(server_url: str, workload: Workload, ...) -> BenchResult: ...

def benchmark(
    *,
    server_url: str,           # http://localhost:8000
    workload: Workload,
    recipe_id: str = "",
    timeout_s: int = 300,
) -> BenchResult:
    """Replay workload against the running server. Returns structured perf+accuracy."""
```

Requests are dispatched concurrently up to `workload.concurrency`
in-flight using `asyncio` + `httpx.AsyncClient`. Streaming
(`stream=True`) is used so TTFT is measured from the first SSE chunk.

### `runner.py`

```python
class VLLMRunner:
    def __init__(self, recipe: Recipe, work_dir: Path,
                 startup_timeout_s: int = 600): ...

    def start(self) -> None:
        """Boot vLLM subprocess. Block until /health returns 200."""
    def stop(self) -> None:
        """SIGTERM + drain. Wait for process exit."""
    def url(self) -> str:
        """Return the OpenAI-compatible server URL."""
    def __enter__(self): self.start(); return self
    def __exit__(self, *_): self.stop()
```

### `llm.py`

Same `provider/model` string convention as `aevyra-origin` and
`aevyra-reflex`:

```python
LLMFn = Callable[[str], str]

def resolve_llm(provider_model: str) -> LLMFn:
    """Parse "anthropic/claude-sonnet-4-6", "openai/gpt-4o", "ollama/qwen3:8b", etc."""
```

### `playbook.py`

```python
@dataclass
class Playbook:
    raw_markdown: str
    sections: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

def load_playbook(path: str | Path) -> Playbook:
    """Parse ## headings into sections. Parse YAML front-matter."""

def format_for_agent(playbook: Playbook, hardware: HardwareSpec, layer: str) -> str:
    """Filter to sections relevant for this hardware + layer.

    Always includes: Search space, Heuristics, Forbidden, Termination.
    Also includes: Hardware: {vendor} if present.
    """
```

---

## v0 scope

### Done ✅

1. `Recipe`, `BenchResult`, `Experiment`, `Workload`, `Playbook` dataclasses
   with YAML/JSON (de)serialization.
2. `runner.VLLMRunner` — subprocess-based vLLM lifecycle.
3. `bench.benchmark` — async concurrent workload replay, streaming TTFT.
4. `workload_from_jsonl`, synthetic generators — with concurrency param
   and path metadata for zero-arg resume.
5. `agent.propose_next_experiment` — single LLM call, structured JSON.
6. `orchestrator.Orchestrator` — main loop + `resume()`.
7. `result.ForgeStore` + `ForgeRun` — run persistence, multi-run
   directory, resume detection.
8. `cli.main` — `aevyra-forge tune/resume/report/playbook`; `--device`
   with auto-detection via `nvidia-smi`/`rocm-smi`.
9. `playbooks/default.md` + bundled `src/aevyra_forge/playbook.md`.
10. `examples/sample_workload.jsonl` — 50-example starter workload.
11. `tests/` — 79 unit tests (workload, recipe, result, playbook, cli).
12. CI — lint (ruff + mypy) + unit tests on Python 3.10/3.11/3.12.

### Out of scope for v0

- Layer 2 (`quant.py` raises `NotImplementedError`)
- Layer 3 (`kernel.py` raises `NotImplementedError`)
- Engines other than vLLM
- Reflex hook to optimize the playbook
- Web dashboard / UI
- Multi-node / disaggregated serving tuning
- Real Langfuse/OTel parsers (stub delegates to JSONL)

---

## Conventions (match Origin)

- **Apache 2.0 license header** on every `.py` file.
- **`from __future__ import annotations`** at the top of every module.
- **Type hints everywhere.** `mypy --strict` clean is the goal.
- **Logging** via `logging.getLogger(__name__)`. No print statements
  in library code.
- **Dataclasses** for every structured value.
- **Private names** start with `_`.
- **Errors** are specific — include what was expected, what was
  received, and (when relevant) a truncated snippet of the offending
  input.

---

## Aevyra integration

Forge composes with the rest of the stack, but does **not** depend on
it at runtime.

- **Witness**: every agent call, every bench run can be captured as a
  span. `Forge.run()` returns the `AgentTrace` alongside the final
  recipe.
- **Origin**: when an experiment fails (CRASH, regression, accuracy
  below floor), the captured trace can be handed to Origin to find
  which decision in the agent's rationale was the root cause.
- **Reflex**: in v0.2+, Reflex consumes the experiment history (the
  agent's predictions vs. actual outcomes) and proposes playbook edits.
  The playbook IS the optimization target.
- **Verdict**: optional accuracy judge for chat workloads.
  lm-eval-harness for math/code is the default.

---

## Dependencies

Runtime (target ≤ 6):

- `aevyra-witness>=0.1.0` — trace the agent loop
- `pyyaml` — recipe serialization
- `httpx` — talk to the running vLLM server
- `anthropic>=0.30` — default agent LLM backend

Optional extras:

- `[openai]` — OpenRouter / OpenAI-compat agent backend
- `[vllm]` — vLLM is the v0 target engine, installed by the user
- `[amd]` — extras for ROCm targets (Quark, AITER) — placeholder
- `[dev]` — pytest, ruff, mypy, typer

**Keep the core dependency-light.** Forge should `pip install` cleanly
in a Colab notebook in under 30 seconds.

---

## Development

```bash
pip install -e ".[dev]"

# Unit tests (no GPU needed)
pytest tests/ -v

# Lint
ruff check . && ruff format --check .

# Dry-run on CPU (no vLLM, synthetic bench results)
aevyra-forge tune \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --device cpu \
  --workload examples/sample_workload.jsonl \
  --dry-run \
  --max-experiments 3

# Full run on a real GPU (Colab T4)
aevyra-forge tune \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --device cuda \
  --workload examples/sample_workload.jsonl \
  --max-experiments 10

# Resume after interruption
aevyra-forge tune resume
```

---

## Contributing

This repo is being built collaboratively. Read these before opening a
PR:

- **AGENT.md is the spec.** If you find yourself building something
  that isn't here, open an issue first. The architecture is meant to
  evolve, but deliberately.
- **Public API surface is small and load-bearing.** Adding a new
  public name is a deliberate decision — document it in this file
  before exporting it from `__init__.py`.
- **The playbook is content, not code.** Edits to `playbooks/*.md`
  are reviewable independently of code changes. Treat them like
  prompt-engineering changes — small commits, before/after benchmark
  numbers in the PR.
- **Vendor-neutrality is a hard rule.** Any new code path that only
  works on one vendor needs a parallel implementation (or a clear
  fallback) for the others, or an explicit gate. Don't merge
  NVIDIA-only paths without an AMD plan in the same PR.
- **Tests are required.** GPU-dependent paths (bench, runner,
  orchestrator with real vLLM) are tested in Colab. Everything else
  must have a unit test in `tests/`.
- **Don't break the Colab quickstart.** It's the first thing every
  evaluator runs. Pin package versions so notebooks don't bitrot.

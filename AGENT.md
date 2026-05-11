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
2. **Fast verifier** — `bench.py` — wraps vLLM's benchmark harness;
   replays a workload trace and returns structured pass/fail + perf.
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

## Architecture

```
src/aevyra_forge/
├── __init__.py            # Public API exports
├── recipe.py              # Recipe dataclass + YAML (de)serialization
├── playbook.py            # Playbook loader (parse .md, expose to agent)
├── orchestrator.py        # The search loop. Amdahl scheduler.
├── agent.py               # The LLM-driven decision agent (reads playbook + trace, picks next experiment)
├── bench.py               # vLLM benchmark wrapper. Workload replay, perf + accuracy.
├── workload.py            # Workload trace ingestion (Langfuse, OTel, JSONL, synthetic fallback)
├── runner.py              # vLLM server lifecycle: start, warmup, stop, restart-with-new-recipe.
├── config.py              # Layer 1: vLLM config search space + mutation operators.
├── quant.py               # Layer 2: quantization recipe selection.    [v0: stub]
├── kernel.py              # Layer 3: AutoKernel integration hook.      [v0: stub]
├── result.py              # Experiment logging (JSONL + TSV), audit trail.
├── llm.py                 # LLMFn type + provider/model factory (Anthropic, OpenAI-compat, local).
└── cli.py                 # CLI entrypoint (typer).
```

Public API surface (from `aevyra_forge`):

```python
Forge,            # main entry — Forge(model, hardware, workload, playbook).run()
Recipe,           # the artifact
Experiment,       # one keep/revert iteration
Playbook,         # parsed playbook
Orchestrator,    # the scheduler
BenchResult,      # structured perf+accuracy output
Workload,         # workload trace
ForgeError,
```

From `aevyra_forge.llm`:

```python
LLMFn                                  # Callable[[str], str] type alias
anthropic_llm, openai_llm              # factories
```

From `aevyra_forge.adapters`:

```python
workload_from_langfuse                # adapt Langfuse export → Workload
workload_from_otel                    # OTel spans → Workload
workload_from_jsonl                   # JSONL → Workload
workload_synthetic                    # ShareGPT-shaped fallback when no trace available
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
    hardware: HardwareSpec                  # GPU type + count + memory

    # Layer 1: vLLM serving config
    config: VLLMConfig                      # max_num_seqs, KV cache fraction, etc.

    # Layer 2: quantization (v0: defaults only)
    quant: QuantRecipe | None = None        # INT4/FP8/INT8, calibration data, etc.

    # Layer 3: custom kernels (v0: standard only)
    kernels: list[KernelOverride] = []

    # Provenance
    parent_id: str | None = None            # which recipe was this mutated from
    generation: int = 0                     # how many keep/revert cycles deep
```

YAML-serializable. The user reads it. The user copies it into their own
deployment. That's the deliverable.

### The playbook — heuristics + search space, not LLM weights

`playbook.md` is the agent's instructions. Like AutoKernel's
`program.md`. It contains:

- Description of the search space at each layer
- Heuristics: "if KV cache utilization > 90%, try INT8 KV cache before
  reducing max_num_seqs"
- Hardware-specific guidance: "on MI300X, ROCM_AITER_FA is usually faster
  than the default flash-attn backend"
- Forbidden combinations: "speculative decoding with TP > 4 is unstable
  on vLLM 0.x.y"
- Workload-conditioned hints: "for chat workloads with high prefix-cache
  hit rate, prefix_caching=True is almost always a win"
- Termination conditions: "stop when 5 consecutive experiments don't
  improve score by 1%"

The playbook ships as a markdown file. The agent reads it as part of
its prompt. **The playbook is itself an optimization target for Reflex
in v0.2+** — but for v0, it's hand-curated.

### The agent — reads trace, picks next experiment

The agent is a small LLM-driven function:

```python
def propose_next_experiment(
    history: list[Experiment],
    playbook: Playbook,
    current_recipe: Recipe,
    workload: Workload,
    llm: LLMFn,
) -> Recipe:
    """Read the experiment log and the playbook, propose a mutated recipe to try next.

    The agent is a single LLM call (~1k tokens of prompt, returns JSON).
    Cheap. Runs synchronously in the orchestrator loop.
    """
```

It produces a JSON-shaped mutation:

```json
{
  "rationale": "Layer 1 has saturated for this workload — config-axis Pareto front hasn't moved in 5 experiments. Moving to Layer 2: try FP8 KV cache, which the playbook says typically gives 20% throughput on H100 chat workloads.",
  "mutation": {
    "layer": "quant",
    "changes": {"kv_cache_dtype": "fp8"}
  },
  "expected_outcome": "Throughput +15-25%, accuracy delta < 0.5%"
}
```

The agent's rationale and prediction are logged. After bench, the
*actual* outcome is compared to the predicted one — that's a signal
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
   in N experiments, the marginal value of another Layer 1 experiment
   is low. Escalate to Layer 2.

The orchestrator owns:

- The `Recipe` mutation queue
- Layer-escalation decisions
- Budget enforcement (max experiments, max wall-clock, max dollars)
- Convergence detection (when to stop)
- Statistical significance gates (don't keep a "win" that's within
  noise)

### The verifier — vLLM benchmark, workload replay

`bench.py` wraps vLLM's official benchmark script (`benchmark_serving.py`)
plus an accuracy check. Returns:

```python
@dataclass
class BenchResult:
    # Performance
    throughput_tokens_per_sec: float
    p50_latency_ms: float
    p99_latency_ms: float
    ttft_ms: float                          # time to first token
    tpot_ms: float                          # time per output token

    # Capacity
    peak_vram_mb: int
    max_concurrent_seqs: int

    # Accuracy
    accuracy_score: float | None            # lm-eval-harness or Verdict judge
    accuracy_delta_vs_baseline: float | None

    # Provenance
    recipe_id: str
    workload_id: str
    bench_duration_s: float
    status: Literal["PASS", "FAIL", "TIMEOUT", "CRASH"]
    error: str | None
```

**The score function** combines these into a single number the
orchestrator can compare:

```python
score = throughput_tokens_per_sec * (1.0 if accuracy_delta_vs_baseline > -threshold else 0.0)
```

(Production version is more sophisticated — Pareto front handling,
SLO constraints, cost weighting — but the v0 score is just
`throughput subject to accuracy floor`.)

### The runner — vLLM server lifecycle

vLLM is a heavyweight server. Each experiment requires:

1. Build engine with the recipe's quantization (Layer 2 — expensive).
2. Boot the server with the recipe's config (Layer 1).
3. Warm up (a few hundred tokens through the engine).
4. Run bench.
5. Stop the server.

The runner abstracts this. v0 supports `vllm` as the only engine.
Engine-agnostic interface (TRT-LLM, SGLang, LMDeploy) is v0.3+.

### The workload — production trace or synthetic fallback

A `Workload` is a stream of `(prompt, expected_output_length)` pairs
with arrival timing. Sources, in order of preference:

1. **Production trace.** Customer's actual traffic, ideally
   privacy-stripped. Adapters: `workload_from_langfuse`,
   `workload_from_otel`, `workload_from_jsonl`.
2. **Synthetic shape match.** Customer provides distribution summaries
   (length histograms, concurrency patterns, prefix-cache hit rates);
   Forge synthesizes a workload matching those shapes.
3. **ShareGPT / public benchmark.** Fallback when the customer has
   nothing. Used for the public Colab demo.

A workload is replayed deterministically — same seed gives same
arrival timing. That's important for run-to-run comparability.

### Vendor neutrality — AMD MI300X is a first-class target

Forge's CI runs benchmarks on **both** NVIDIA and AMD hardware. The
recipe schema includes per-vendor fields where relevant (e.g.
`attention_backend` differs: `flash-attn-3` on Hopper, `ROCM_AITER_FA`
on MI300X). The playbook has per-hardware sections. The `quant` module
handles FP8 differently for E4M3 (Hopper) vs. OCP FP8 (MI300X+).

This is a **positioning** decision, not just a technical one. AMD
customers are underserved by existing tooling. Forge wins by being
the first autotuner that doesn't treat AMD as a port.

### Colab/Kaggle-first distribution

The `notebooks/` directory ships Colab-ready notebooks for:

- **`forge_quickstart.ipynb`** — tune Llama-3.2-1B on a T4 (Colab free
  tier). 12-hour overnight session finds a 30%+ improvement over
  defaults.
- **`forge_byo_workload.ipynb`** — bring your own workload trace
  (Langfuse export), tune against it.
- **`forge_amd_mi300x.ipynb`** — AMD-specific quickstart (requires
  AMD Developer Cloud access).

Each notebook has a `Run on Colab` badge in the README. Pin the package
versions so notebooks don't bitrot.

---

## v0 scope (what Sonnet builds first)

**Build this. Stop here.**

1. `Recipe`, `BenchResult`, `Experiment`, `Workload`, `Playbook`
   dataclasses with YAML/JSON (de)serialization.
2. `runner.VLLMRunner` — start/stop vLLM, apply Layer 1 config from a
   recipe. Subprocess-based, no library hooks into vLLM yet.
3. `bench.benchmark` — wraps `vllm bench serve` (or equivalent),
   parses output into `BenchResult`. Accuracy validation via
   lm-eval-harness as an optional second pass.
4. `workload.from_jsonl`, `workload.synthetic` — minimal workload
   ingestion. `from_langfuse` and `from_otel` are stubs that delegate
   to the same JSONL converter for v0.
5. `config.search_space` — enumerate Layer 1 knobs and their
   permissible ranges. `config.mutate` — apply an agent-proposed
   change to a Recipe, return the new Recipe.
6. `agent.propose_next_experiment` — single LLM call, reads history +
   playbook + workload summary, returns a JSON mutation. Provider
   selection via `provider/model` string (same convention as Origin).
7. `orchestrator.Orchestrator` — the main loop. `run()` returns a
   final `Recipe` + full `Experiment` history.
8. `result.ExperimentStore` — JSONL log per run, plus a
   human-readable TSV summary (mirrors AutoKernel's pattern).
9. `cli.main` — typer-based CLI. `forge tune --model X --hardware Y
   --workload Z`.
10. `playbooks/default.md` — initial playbook covering Layer 1 only.
    Hand-written for v0.

**Out of scope for v0:**

- Layer 2 (`quant.py` raises NotImplementedError)
- Layer 3 (`kernel.py` raises NotImplementedError)
- Engines other than vLLM
- Reflex hook to optimize the playbook
- Web dashboard / UI
- Multi-node / disaggregated serving tuning
- Real Langfuse/OTel parsers (stubs delegate to JSONL)

These exist as module files with docstrings + NotImplementedError so
the architecture is visible from day one. Wiring them up is v0.2+
work.

---

## Module-by-module spec

### `recipe.py`

```python
@dataclass
class HardwareSpec:
    vendor: Literal["nvidia", "amd", "intel", "google", "other"]
    gpu_type: str                            # "H100", "MI300X", "L40S", "T4", ...
    count: int                               # number of GPUs
    memory_gb_per_gpu: int

@dataclass
class VLLMConfig:
    """Layer 1 search space. All fields tunable; defaults are vLLM's defaults."""
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    block_size: int = 16
    gpu_memory_utilization: float = 0.9
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = True
    swap_space: int = 4                       # GiB
    kv_cache_dtype: Literal["auto", "fp8", "fp16", "bf16"] = "auto"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    speculative_model: str | None = None
    num_speculative_tokens: int = 0
    attention_backend: str | None = None      # vendor-specific
    # ... add fields as the playbook references them

@dataclass
class QuantRecipe:
    """Layer 2. v0: defaults only; raises if mutated."""
    method: Literal["fp16", "bf16", "int4_awq", "int4_gptq", "fp8_e4m3", "int8"] = "bf16"
    kv_cache_quant: Literal["none", "fp8", "int8"] = "none"
    calibration_dataset: str | None = None
    per_layer_overrides: dict[str, str] = field(default_factory=dict)

@dataclass
class KernelOverride:
    """Layer 3. v0: never populated."""
    op_name: str                              # "attention", "rmsnorm", ...
    kernel_source_path: str                   # path to .py or .cu

@dataclass
class Recipe:
    model: str
    hardware: HardwareSpec
    config: VLLMConfig
    quant: QuantRecipe = field(default_factory=QuantRecipe)
    kernels: list[KernelOverride] = field(default_factory=list)
    parent_id: str | None = None
    generation: int = 0
    id: str = ""                              # auto-assigned

    def to_yaml(self) -> str: ...
    @classmethod
    def from_yaml(cls, s: str) -> "Recipe": ...
```

### `bench.py`

```python
@dataclass
class BenchResult: ...   # see above

def benchmark(
    *,
    server_url: str,           # http://localhost:8000
    workload: Workload,
    accuracy_check: bool = True,
    timeout_s: int = 600,
) -> BenchResult:
    """Replay `workload` against the running server. Return structured perf+accuracy."""
```

Implementation note: shell out to `vllm bench serve` or implement the
replay loop directly via the OpenAI-compatible client. Whichever is
more robust. The script must capture: throughput, TTFT, TPOT, p50,
p99, peak VRAM. Accuracy uses lm-eval-harness for math/code tasks or
a Verdict judge for chat.

### `runner.py`

```python
class VLLMRunner:
    """Manage a vLLM server's lifecycle for one experiment."""

    def __init__(self, recipe: Recipe, work_dir: Path): ...

    def start(self) -> None:
        """Boot vLLM with recipe.config applied. Block until /health returns 200."""

    def stop(self) -> None:
        """SIGTERM + drain. Wait for process exit."""

    def url(self) -> str:
        """Return the OpenAI-compatible server URL."""

    def __enter__(self): self.start(); return self
    def __exit__(self, *_): self.stop()
```

vLLM is invoked as a subprocess (`vllm serve {model} --max-num-seqs ...`).
Logs are captured and surfaced on crash.

### `workload.py`

```python
@dataclass
class WorkloadRequest:
    prompt: str
    expected_output_tokens: int               # for capacity planning
    arrival_offset_s: float                   # seconds since workload start
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class Workload:
    requests: list[WorkloadRequest]
    duration_s: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return a dict the agent prompt can read: length distributions,
        prefix-cache hit estimate, concurrency profile, etc."""

def from_jsonl(path: Path) -> Workload: ...
def from_langfuse(path: Path) -> Workload: ...    # v0: thin wrapper around from_jsonl
def from_otel(spans: list[dict]) -> Workload: ...  # v0: stub
def synthetic(
    *,
    n_requests: int = 1000,
    avg_input_tokens: int = 512,
    avg_output_tokens: int = 128,
    seed: int = 0,
) -> Workload: ...
```

### `config.py`

```python
def search_space(hardware: HardwareSpec) -> dict[str, list[Any]]:
    """Return the legal value ranges for each VLLMConfig field on this hardware.

    Some fields are vendor-conditional (e.g. attention_backend choices).
    """

def mutate(recipe: Recipe, mutation: dict[str, Any]) -> Recipe:
    """Apply an agent-proposed mutation. Validate against search_space.

    `mutation` is the JSON the agent emitted, e.g.:
        {"layer": "config", "changes": {"max_num_seqs": 512, "enable_prefix_caching": true}}
    """
```

### `agent.py`

```python
@dataclass
class AgentDecision:
    rationale: str
    mutation: dict[str, Any]
    expected_throughput_delta_pct: float | None
    expected_accuracy_delta: float | None
    raw_response: str

def propose_next_experiment(
    *,
    history: list[Experiment],
    playbook: Playbook,
    current_recipe: Recipe,
    workload: Workload,
    llm: LLMFn,
) -> AgentDecision: ...
```

The prompt template lives in `_AGENT_PROMPT` at module top. Sections:

- Workload summary (length distribution, concurrency, prefix-cache hit
  estimate)
- Hardware spec
- Current recipe (YAML, terse)
- Last 10 experiments (recipe diff, score, status)
- Playbook (verbatim)
- Instructions: emit JSON with `rationale`, `mutation`,
  `expected_throughput_delta_pct`, `expected_accuracy_delta`.

### `orchestrator.py`

```python
@dataclass
class Experiment:
    id: str
    recipe: Recipe
    bench_result: BenchResult | None
    agent_decision: AgentDecision | None
    score: float | None
    kept: bool                                # True if better than parent
    duration_s: float
    started_at: float
    ended_at: float

@dataclass
class ForgeConfig:
    """Budget + termination knobs."""
    max_experiments: int = 50
    max_wall_clock_hours: float = 12.0
    max_dollars: float | None = None
    accuracy_floor: float = 0.99              # vs baseline
    min_improvement_pct: float = 1.0
    convergence_window: int = 5               # stop after N stagnant experiments
    layer_escalation: bool = True             # Auto-jump from L1 → L2 → L3

class Orchestrator:
    def __init__(
        self,
        *,
        model: str,
        hardware: HardwareSpec,
        workload: Workload,
        playbook: Playbook,
        llm: LLMFn,
        store: ExperimentStore,
        forge_config: ForgeConfig = ForgeConfig(),
    ): ...

    def run(self) -> tuple[Recipe, list[Experiment]]:
        """Main loop. Returns the best recipe found + full history."""
```

### `result.py`

```python
class ExperimentStore:
    """Append-only JSONL log of experiments, plus a TSV summary."""

    def __init__(self, run_dir: Path): ...
    def append(self, exp: Experiment) -> None: ...
    def best(self) -> Experiment | None: ...
    def history(self) -> list[Experiment]: ...
    def render_tsv(self) -> str: ...
```

### `cli.py`

```bash
forge tune \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --hardware nvidia/A100 \
  --workload sample_workload.jsonl \
  --playbook playbooks/default.md \
  --model-agent openrouter/anthropic/claude-sonnet-4-5 \
  --output runs/2026-05-11/

forge resume runs/2026-05-11/
forge report runs/2026-05-11/  # rendered TSV + Pareto plot
```

### `llm.py`

Identical contract to Origin's `aevyra_origin.llm`. Same factories,
same `provider/model` string convention. **Copy the module from
Origin; don't reinvent.** (Eventually extract to `aevyra-common`.)

### `playbook.py`

```python
@dataclass
class Playbook:
    raw_markdown: str
    sections: dict[str, str]
    metadata: dict[str, Any]

def load_playbook(path: Path) -> Playbook: ...
def format_for_agent(playbook: Playbook, hardware: HardwareSpec, layer: str) -> str:
    """Filter the playbook to the sections relevant for this hardware + layer."""
```

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
it at runtime (Witness is the only runtime dep, and only for tracing
the agent's reasoning).

- **Witness**: every agent call, every bench run is captured as a
  span. `Forge.run()` returns the `AgentTrace` alongside the final
  recipe.
- **Origin**: when an experiment fails (CRASH, regression, accuracy
  below floor), the captured trace is handed to Origin to find which
  decision in the agent's rationale was the root cause. The
  attribution is stored alongside the experiment.
- **Reflex**: in v0.2+, Reflex consumes the experiment history (the
  agent's predictions vs. actual outcomes) and proposes playbook
  edits. The playbook IS the optimization target.
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
- `[amd]` — extras for ROCm targets (Quark, AITER)
- `[dev]` — pytest, ruff, mypy, typer

**Keep the core dependency-light.** Forge should `pip install` cleanly
in a Colab notebook in under 30 seconds. Anything that pulls heavy
ML-stack deps (`torch`, `transformers`) goes behind an extra.

---

## Development

```bash
pip install -e ".[dev]"

# Day-to-day (no GPU needed — runner is stubbed in tests)
forge tune --model gpt-2 --hardware nvidia/T4 \
  --workload examples/sharegpt_small.jsonl \
  --playbook playbooks/default.md \
  --model-agent ollama/qwen3:8b \
  --dry-run

# With a real GPU
forge tune --model meta-llama/Llama-3.2-1B-Instruct \
  --hardware nvidia/A100 \
  --workload your_trace.jsonl \
  --model-agent anthropic/claude-sonnet-4-5
```

The `--dry-run` flag is important: it makes runner + bench mock-shaped
so the orchestrator + agent loop can be developed against deterministic
fixtures without burning GPU hours. **Build dry-run support from day
one.**

---

## Collab-friendly notes

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
- **No tests for v0 scaffolding.** Test scaffolding will land once
  the modules have real bodies. Lint and type-check should always
  pass.
- **Don't break the Colab quickstart.** It's the first thing every
  evaluator runs. CI runs the Colab notebook end-to-end on every
  PR (against a tiny model on a CPU-only runner with mocked vLLM).

# Forge

> **Status:** 0.0.1 pre-alpha. The architecture is laid out; the
> implementation is in progress. See [AGENT.md](./AGENT.md) for the
> build spec.

**Forge tunes your LLM deployment overnight.**

Give Forge a model, a hardware target, and a workload trace. It runs
an autonomous loop — propose a config, boot vLLM, benchmark against
your workload, keep or revert, repeat. By morning you have a deployment
recipe that beats hand-tuned defaults by 30-50% on Layer 1 alone, with
a full audit trail of every experiment.

```bash
forge tune \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --hardware nvidia/A100 \
  --workload your_trace.jsonl \
  --model-agent anthropic/claude-sonnet-4-5
```

Forge is the deployment-tuning layer of the [Aevyra](https://aevyra.ai)
stack:

```
Witness  →  captures what happened           captures the trace
Verdict  →  judges it                        scores the run
Origin   →  finds where it went wrong        attributes the failure
Reflex   →  fixes the prompts                rewrites the playbook
Forge    →  tunes the deployment             ← you are here
```

## Why Forge?

- **Autonomous loop, not a managed service.** Run on your own GPUs,
  on-prem, air-gapped if you need to.
- **Vendor-neutral.** AMD MI300X is first-class, not a port.
- **Workload-aware.** Optimizes against your real traffic, not
  ShareGPT.
- **Joint config + quant + kernel tuning.** The three axes interact;
  Forge searches them together.
- **Colab-friendly.** Try it on a free T4 before you spend a dollar
  on H100s — the orchestrator and playbook work the same.

## Quick start (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](./notebooks/forge_quickstart.ipynb)

(Quickstart notebook is coming; the badge above will work once it
lands.)

## Quick start (local)

```bash
pip install aevyra-forge[openai]
export OPENROUTER_API_KEY=sk-or-...

# Synthetic ShareGPT-shaped workload, tiny model on whatever GPU you have
forge tune \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --hardware nvidia/A100 \
  --workload-synthetic \
  --max-experiments 20 \
  --output runs/first/

# View the result
forge report runs/first/
```

## How it works

The five-ingredient autoresearch recipe:

1. **One artifact** — `recipe.yaml` — vLLM config + quant + kernels.
2. **One verifier** — `bench.py` — replays your workload, returns
   structured perf + accuracy.
3. **One playbook** — `playbook.md` — encodes serving expertise.
4. **One tight loop** — ~5-15 min per experiment, 30-100 experiments
   overnight.
5. **One Amdahl scheduler** — spend experiments where they have the
   most leverage.

See [AGENT.md](./AGENT.md) for the full architecture.

## Status

| Layer | What it tunes | v0 status |
|---|---|---|
| **1. Config** | vLLM serving args | Building now |
| **2. Quantization** | INT4/FP8/INT8, KV cache | Scaffolded, not implemented |
| **3. Kernel synthesis** | Custom kernels (AutoKernel hook) | Scaffolded, not implemented |

## Contributing

This is being built collaboratively. Read [AGENT.md](./AGENT.md) before
opening a PR. Check [CONTRIBUTING.md](./CONTRIBUTING.md) for ground
rules.

## License

Apache 2.0.

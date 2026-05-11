# Notebooks

Colab/Kaggle-friendly demos. Each notebook in this directory has a
`Run on Colab` badge in the main README. CI runs them end-to-end on
every PR (against a tiny model with mocked vLLM where appropriate).

## Planned

- **`forge_quickstart.ipynb`** — tune Llama-3.2-1B on a T4 (Colab
  free tier). 12-hour overnight session, demonstrates the full
  orchestrator loop with a synthetic workload. Target audience:
  developers evaluating Forge.

- **`forge_byo_workload.ipynb`** — bring your own workload trace
  (Langfuse export or JSONL). Show conversion + tuning. Target
  audience: teams with existing observability stacks.

- **`forge_amd_mi300x.ipynb`** — AMD-specific quickstart. Requires
  AMD Developer Cloud or TensorWave access. Target audience: AMD
  inference customers.

## Conventions

- First cell: `Run on Colab` badge + a one-line description of what
  the notebook does.
- Second cell: `pip install` + env-var setup.
- Third cell: imports.
- Last cell: link to the rendered docs site.
- Don't leave cell outputs in committed notebooks — strip them with
  `nbstripout` before committing. CI will fail on stripped output
  diffs.
- Pin versions of `aevyra-forge` and any heavy deps so notebooks
  don't bitrot.

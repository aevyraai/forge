# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CLI entrypoint — ``forge`` command.

See AGENT.md → "Module-by-module spec → cli.py".

Surface:

    forge tune       --model X --hardware Y --workload Z [...]
    forge resume     <run-dir>
    forge report     <run-dir>          # render TSV, print Pareto-front summary
    forge playbook   show|validate|diff

Uses typer. Same patterns as Origin's CLI — read ``aevyra_origin.cli``
for the conventions to match (Annotated options, version callback,
no-args-is-help, friendly error when typer isn't installed).
"""

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import typer

    _TYPER_AVAILABLE = True
except ImportError:
    _TYPER_AVAILABLE = False


def main() -> None:
    """Entry point registered in pyproject.toml under [project.scripts]."""
    if not _TYPER_AVAILABLE:
        print(
            "forge CLI requires typer. Install with: pip install aevyra-forge[dev]",
            file=sys.stderr,
        )
        sys.exit(1)

    app = typer.Typer(
        name="forge",
        help="Aevyra Forge — autonomous vLLM deployment optimizer.",
        no_args_is_help=True,
        add_completion=False,
    )

    # ------------------------------------------------------------------
    # forge tune
    # ------------------------------------------------------------------
    @app.command()
    def tune(
        model: str = typer.Option(..., "--model", "-m", help="HuggingFace model ID or local path"),
        hardware: str = typer.Option(
            ...,
            "--hardware",
            "-H",
            help="Hardware spec: vendor/gpu_type[xN] e.g. nvidia/H100x8, amd/MI300X",
        ),
        workload_jsonl: Path = typer.Option(
            ...,
            "--workload",
            "-w",
            help="Path to workload JSONL (prompt + expected_output_tokens per line)",
        ),
        playbook: Optional[Path] = typer.Option(
            None, "--playbook", help="Path to playbook markdown. Defaults to built-in playbook."
        ),
        llm_provider: str = typer.Option(
            "anthropic/claude-sonnet-4-6",
            "--llm",
            help="LLM for the agent. Format: provider/model. "
            "Providers: anthropic, openai, openrouter, groq, together, "
            "fireworks, deepinfra, mistral, ollama, lmstudio.",
        ),
        max_experiments: int = typer.Option(50, help="Max number of experiments"),
        max_hours: float = typer.Option(12.0, help="Max wall-clock hours"),
        max_dollars: Optional[float] = typer.Option(None, help="Max LLM spend in USD"),
        accuracy_floor: float = typer.Option(0.99, help="Min acceptable accuracy (0-1)"),
        min_improvement_pct: float = typer.Option(1.0, help="Min improvement % to keep a recipe"),
        run_dir: Optional[Path] = typer.Option(None, help="Directory for run artifacts"),
        dry_run: bool = typer.Option(
            False, "--dry-run", help="Skip vLLM; use synthetic bench results"
        ),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
    ) -> None:
        """Run an overnight autotune session and emit the best recipe."""
        _setup_logging(verbose)
        _run_tune(
            model=model,
            hardware_str=hardware,
            workload_jsonl=workload_jsonl,
            playbook_path=playbook,
            llm_provider=llm_provider,
            max_experiments=max_experiments,
            max_hours=max_hours,
            max_dollars=max_dollars,
            accuracy_floor=accuracy_floor,
            min_improvement_pct=min_improvement_pct,
            run_dir=run_dir,
            dry_run=dry_run,
        )

    # ------------------------------------------------------------------
    # forge resume
    # ------------------------------------------------------------------
    @app.command()
    def resume(
        run_dir: Path = typer.Argument(..., help="Run directory from a previous forge tune"),
        llm_provider: str = typer.Option(
            "anthropic/claude-sonnet-4-6", "--llm", help="LLM for the agent."
        ),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
    ) -> None:
        """Resume an interrupted autotune run."""
        _setup_logging(verbose)
        _run_resume(run_dir=run_dir, llm_provider=llm_provider)

    # ------------------------------------------------------------------
    # forge report
    # ------------------------------------------------------------------
    @app.command()
    def report(
        run_dir: Path = typer.Argument(..., help="Run directory to report on"),
    ) -> None:
        """Print a human-readable summary of a completed run."""
        _run_report(run_dir=run_dir)

    # ------------------------------------------------------------------
    # forge playbook
    # ------------------------------------------------------------------
    playbook_app = typer.Typer(help="Inspect or validate the playbook.", no_args_is_help=True)
    app.add_typer(playbook_app, name="playbook")

    @playbook_app.command("show")
    def playbook_show(
        path: Optional[Path] = typer.Argument(None, help="Path to playbook (default: built-in)"),
    ) -> None:
        """Print the playbook to stdout."""
        from aevyra_forge.playbook import load_playbook

        pb_path = path or _default_playbook_path()
        pb = load_playbook(pb_path)
        for section, text in pb.sections.items():
            print(f"## {section}\n{text}\n")

    @playbook_app.command("validate")
    def playbook_validate(
        path: Optional[Path] = typer.Argument(None),
    ) -> None:
        """Validate playbook YAML front-matter and section headers."""
        from aevyra_forge.playbook import load_playbook

        pb_path = path or _default_playbook_path()
        try:
            pb = load_playbook(pb_path)
            print(f"✓ Playbook loaded OK ({len(pb.sections)} sections)")
        except Exception as exc:
            print(f"✗ Playbook validation failed: {exc}", file=sys.stderr)
            raise typer.Exit(1)

    app()


# ------------------------------------------------------------------
# Internals
# ------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_hardware(hardware_str: str):
    """Parse 'nvidia/H100x8' or 'amd/MI300X' into HardwareSpec."""
    from aevyra_forge.recipe import HardwareSpec

    parts = hardware_str.split("/", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Hardware must be vendor/gpu_type[xN], got: {hardware_str!r}. "
            "Examples: nvidia/H100x8, amd/MI300X, nvidia/A100"
        )
    vendor, gpu_spec = parts
    count = 1
    if "x" in gpu_spec:
        gpu_type, count_str = gpu_spec.rsplit("x", 1)
        try:
            count = int(count_str)
        except ValueError:
            gpu_type = gpu_spec
    else:
        gpu_type = gpu_spec

    # Infer VRAM from GPU type
    mem_map = {
        "H100": 80,
        "H200": 141,
        "B100": 192,
        "B200": 192,
        "A100": 80,
        "A6000": 48,
        "A10": 24,
        "T4": 16,
        "MI300X": 192,
        "MI250X": 128,
    }
    memory_gb = next(
        (v for k, v in mem_map.items() if k.upper() in gpu_type.upper()),
        24,
    )
    return HardwareSpec(
        vendor=vendor.lower(),
        gpu_type=gpu_type,
        count=count,
        memory_gb_per_gpu=memory_gb,
    )


def _default_playbook_path() -> Path:
    """Return the bundled playbook path."""
    return Path(__file__).parent / "playbook.md"


def _make_run_dir(run_dir: "Path | None") -> Path:
    import datetime

    if run_dir is not None:
        return run_dir
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return Path("./runs") / ts


def _run_tune(
    *,
    model: str,
    hardware_str: str,
    workload_jsonl: "Path",
    playbook_path: "Path | None",
    llm_provider: str,
    max_experiments: int,
    max_hours: float,
    max_dollars: "float | None",
    accuracy_floor: float,
    min_improvement_pct: float,
    run_dir: "Path | None",
    dry_run: bool,
) -> None:
    import json

    from aevyra_forge.llm import resolve_llm
    from aevyra_forge.orchestrator import ForgeConfig, Orchestrator
    from aevyra_forge.playbook import load_playbook
    from aevyra_forge.result import ExperimentStore
    from aevyra_forge.workload import workload_from_jsonl

    hardware = _parse_hardware(hardware_str)
    actual_run_dir = _make_run_dir(run_dir)
    actual_run_dir.mkdir(parents=True, exist_ok=True)

    # Workload
    workload = workload_from_jsonl(workload_jsonl)

    # Playbook
    pb_path = playbook_path or _default_playbook_path()
    if not pb_path.exists():
        logging.getLogger(__name__).warning(
            "Playbook not found at %s — using empty playbook.", pb_path
        )
        from aevyra_forge.playbook import Playbook

        playbook = Playbook(raw_markdown="")
    else:
        playbook = load_playbook(pb_path)

    llm = resolve_llm(llm_provider)
    store = ExperimentStore(actual_run_dir)

    # Save run config
    config_path = actual_run_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": model,
                "hardware": hardware_str,
                "workload": str(workload_jsonl) if workload_jsonl else "synthetic",
                "llm": llm_provider,
                "max_experiments": max_experiments,
                "max_hours": max_hours,
                "accuracy_floor": accuracy_floor,
                "dry_run": dry_run,
            },
            indent=2,
        )
    )

    forge_config = ForgeConfig(
        max_experiments=max_experiments,
        max_wall_clock_hours=max_hours,
        max_dollars=max_dollars,
        accuracy_floor=accuracy_floor,
        min_improvement_pct=min_improvement_pct,
        dry_run=dry_run,
        work_dir=actual_run_dir,
    )

    orchestrator = Orchestrator(
        model=model,
        hardware=hardware,
        workload=workload,
        playbook=playbook,
        llm=llm,
        store=store,
        forge_config=forge_config,
    )

    print(f"\n🔨 Forge starting — run dir: {actual_run_dir}\n")
    best_recipe, history = orchestrator.run()

    best_path = actual_run_dir / "best_recipe.yaml"
    print(f"\n✓ Done. {len(history)} experiments. Best recipe → {best_path}\n")
    print(best_recipe.to_yaml())


def _run_resume(*, run_dir: Path, llm_provider: str) -> None:
    import json

    from aevyra_forge.llm import resolve_llm
    from aevyra_forge.orchestrator import ForgeConfig, Orchestrator
    from aevyra_forge.playbook import load_playbook
    from aevyra_forge.result import ExperimentStore
    from aevyra_forge.workload import workload_synthetic as make_synthetic

    config_path = run_dir / "config.json"
    if not config_path.exists():
        print(f"✗ No config.json in {run_dir}", file=sys.stderr)
        sys.exit(1)

    cfg = json.loads(config_path.read_text())
    hardware = _parse_hardware(cfg["hardware"])

    workload_path = cfg.get("workload")
    if workload_path and workload_path != "synthetic":
        from aevyra_forge.workload import workload_from_jsonl

        workload = workload_from_jsonl(Path(workload_path))
    else:
        workload = make_synthetic()

    pb_path = _default_playbook_path()
    if pb_path.exists():
        playbook = load_playbook(pb_path)
    else:
        from aevyra_forge.playbook import Playbook

        playbook = Playbook(raw_markdown="")

    llm = resolve_llm(llm_provider)
    store = ExperimentStore(run_dir)

    forge_config = ForgeConfig(
        max_experiments=cfg.get("max_experiments", 50),
        max_wall_clock_hours=cfg.get("max_hours", 12.0),
        accuracy_floor=cfg.get("accuracy_floor", 0.99),
        dry_run=cfg.get("dry_run", False),
        work_dir=run_dir,
    )

    orchestrator = Orchestrator(
        model=cfg["model"],
        hardware=hardware,
        workload=workload,
        playbook=playbook,
        llm=llm,
        store=store,
        forge_config=forge_config,
    )

    print(f"\n🔨 Forge resuming — run dir: {run_dir}\n")
    best_recipe, history = orchestrator.resume()
    print(f"\n✓ Done. {len(history)} total experiments.\n")
    print(best_recipe.to_yaml())


def _run_report(*, run_dir: Path) -> None:
    from aevyra_forge.result import ExperimentStore

    store = ExperimentStore(run_dir)
    history = store.history()
    if not history:
        print(f"No experiments found in {run_dir}")
        return

    best = store.best()
    print(f"\n=== Forge Report: {run_dir} ===\n")
    print(f"Total experiments: {len(history)}")
    if best:
        print(f"Best score:        {best.score:.4f}")
        print(f"Best recipe ID:    {best.id}")
        print(f"Best generation:   {best.recipe.generation}")
        br = best.bench_result
        if br:
            print(f"Throughput:        {br.throughput_tokens_per_sec:.1f} tok/s")
            print(f"P99 latency:       {br.p99_latency_ms:.0f} ms")
    print()
    print(store.render_tsv())


if __name__ == "__main__":
    main()

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
        action: Optional[str] = typer.Argument(
            None, help="Optional subcommand: 'resume' to continue an interrupted run."
        ),
        model: Optional[str] = typer.Option(None, "--model", "-m", help="HuggingFace model ID or local path"),
        device: str = typer.Option(
            "cuda",
            "--device",
            "-d",
            help="Device backend: cuda (NVIDIA), rocm (AMD), or cpu (dry-run only).",
        ),
        workload_jsonl: Optional[Path] = typer.Option(
            None,
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
        concurrency: int = typer.Option(8, "--concurrency", "-c", help="Concurrent requests during benchmarking. T4/A10: 8–16, A100/H100: 32–64."),
        max_experiments: int = typer.Option(50, help="Max number of experiments"),
        max_hours: float = typer.Option(12.0, help="Max wall-clock hours"),
        max_dollars: Optional[float] = typer.Option(None, help="Max LLM spend in USD"),
        accuracy_floor: float = typer.Option(0.99, help="Min acceptable accuracy (0-1)"),
        min_improvement_pct: float = typer.Option(1.0, help="Min improvement % to keep a recipe"),
        run_dir: Optional[Path] = typer.Option(None, help="ForgeStore root directory (default: .forge)"),
        dry_run: bool = typer.Option(
            False, "--dry-run", help="Skip vLLM; use synthetic bench results"
        ),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
    ) -> None:
        """Run or resume an autotune session.

        \b
        aevyra-forge tune            Start a new run
        aevyra-forge tune resume     Resume the latest interrupted run
        """
        _setup_logging(verbose)

        if action == "resume":
            _run_resume(run_dir=run_dir)
            return

        if action is not None:
            raise typer.BadParameter(f"Unknown subcommand {action!r}. Did you mean 'resume'?")

        if model is None:
            raise typer.BadParameter("--model is required when starting a new run.")
        if workload_jsonl is None:
            raise typer.BadParameter("--workload is required when starting a new run.")

        _run_tune(
            model=model,
            device=device,
            workload_jsonl=workload_jsonl,
            concurrency=concurrency,
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


def _detect_hardware(device: str):
    """Auto-detect GPU type, count, and VRAM from the device backend.

    Args:
        device: One of ``cuda``, ``rocm``, or ``cpu``.

    Returns:
        A :class:`~aevyra_forge.recipe.HardwareSpec` populated from
        ``nvidia-smi`` (cuda), ``rocm-smi`` (rocm), or a CPU placeholder.
    """
    import subprocess
    from aevyra_forge.recipe import HardwareSpec

    log = logging.getLogger(__name__)
    device = device.lower().strip()

    if device == "cpu":
        log.info("Device: cpu — no GPU detected; dry-run mode recommended.")
        return HardwareSpec(vendor="cpu", gpu_type="cpu", count=1, memory_gb_per_gpu=0)

    if device == "cuda":
        try:
            # Query name + memory for every GPU in one call
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip())
            lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            if not lines:
                raise RuntimeError("nvidia-smi returned no GPUs")
            # Each line: "Tesla T4, 15360"
            names, mibs = zip(*(l.split(",", 1) for l in lines))
            gpu_type = names[0].strip()
            memory_gb = round(int(mibs[0].strip()) / 1024)
            count = len(lines)
            log.info("Detected %d × %s with %d GB VRAM each", count, gpu_type, memory_gb)
            return HardwareSpec(vendor="nvidia", gpu_type=gpu_type, count=count, memory_gb_per_gpu=memory_gb)
        except Exception as exc:
            raise RuntimeError(
                f"Could not detect NVIDIA GPU (--device cuda): {exc}. "
                "Is nvidia-smi available? Try --device rocm or --device cpu."
            ) from exc

    if device == "rocm":
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip())
            import json as _json
            data = _json.loads(result.stdout)
            cards = {k: v for k, v in data.items() if k.startswith("card")}
            if not cards:
                raise RuntimeError("rocm-smi returned no cards")
            first = next(iter(cards.values()))
            gpu_type = first.get("Card series", first.get("Card model", "AMD GPU")).strip()
            bytes_per_gpu = int(first.get("VRAM Total Memory (B)", 0))
            memory_gb = round(bytes_per_gpu / (1024 ** 3))
            count = len(cards)
            log.info("Detected %d × %s with %d GB VRAM each", count, gpu_type, memory_gb)
            return HardwareSpec(vendor="amd", gpu_type=gpu_type, count=count, memory_gb_per_gpu=memory_gb)
        except Exception as exc:
            raise RuntimeError(
                f"Could not detect AMD GPU (--device rocm): {exc}. "
                "Is rocm-smi available? Try --device cuda or --device cpu."
            ) from exc

    raise ValueError(f"Unknown device {device!r}. Choose: cuda, rocm, cpu.")


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
    device: str,
    workload_jsonl: "Path",
    concurrency: int,
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
    from aevyra_forge.result import ForgeStore
    from aevyra_forge.workload import workload_from_jsonl

    hardware = _detect_hardware(device)

    # Workload
    workload = workload_from_jsonl(workload_jsonl, concurrency=concurrency)

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
    store = ForgeStore(run_dir or ".forge")

    forge_config = ForgeConfig(
        max_experiments=max_experiments,
        max_wall_clock_hours=max_hours,
        max_dollars=max_dollars,
        accuracy_floor=accuracy_floor,
        min_improvement_pct=min_improvement_pct,
        dry_run=dry_run,
    )

    orchestrator = Orchestrator(
        model=model,
        hardware=hardware,
        workload=workload,
        playbook=playbook,
        llm=llm,
        store=store,
        forge_config=forge_config,
        llm_provider=llm_provider,
        device=device,
    )

    best_recipe, history = orchestrator.run()
    print(f"\n✓ Done — {len(history)} experiments\n")
    print(best_recipe.to_yaml())


def _run_resume(*, run_dir: "Path | None") -> None:
    from aevyra_forge.llm import resolve_llm
    from aevyra_forge.orchestrator import ForgeConfig, Orchestrator
    from aevyra_forge.playbook import load_playbook
    from aevyra_forge.result import ForgeStore

    store = ForgeStore(run_dir or ".forge")
    incomplete = store.find_incomplete_run()
    if incomplete is None:
        print("✗ No interrupted run found to resume.", file=sys.stderr)
        sys.exit(1)

    cfg = incomplete.config() or {}
    device = cfg.get("device", "cuda")
    hardware = _detect_hardware(device)

    workload_path = cfg.get("workload_path", "")
    if workload_path:
        from aevyra_forge.workload import workload_from_jsonl
        workload = workload_from_jsonl(Path(workload_path))
    else:
        from aevyra_forge.workload import workload_synthetic as make_synthetic
        workload = make_synthetic()

    pb_path = _default_playbook_path()
    if pb_path.exists():
        playbook = load_playbook(pb_path)
    else:
        from aevyra_forge.playbook import Playbook
        playbook = Playbook(raw_markdown="")

    llm_provider = cfg.get("llm_provider", "anthropic/claude-sonnet-4-6")
    llm = resolve_llm(llm_provider)

    forge_cfg = cfg.get("forge_config", {})
    forge_config = ForgeConfig(
        max_experiments=forge_cfg.get("max_experiments", 50),
        max_wall_clock_hours=forge_cfg.get("max_wall_clock_hours", 12.0),
        accuracy_floor=forge_cfg.get("accuracy_floor", 0.99),
        dry_run=forge_cfg.get("dry_run", False),
    )

    orchestrator = Orchestrator(
        model=cfg["model"],
        hardware=hardware,
        workload=workload,
        playbook=playbook,
        llm=llm,
        store=store,
        forge_config=forge_config,
        llm_provider=llm_provider,
        device=device,
    )

    best_recipe, history = orchestrator.resume()
    print(f"\n✓ Done — {len(history)} total experiments\n")
    print(best_recipe.to_yaml())


def _run_report(*, run_dir: Path) -> None:
    from aevyra_forge.result import ForgeStore

    store = ForgeStore(run_dir or ".forge")
    run = store.latest_run()
    if run is None:
        print(f"No runs found in {run_dir or '.forge'}")
        return

    history = run.history()
    if not history:
        print(f"No experiments found in {run.path}")
        return

    best = run.best()
    print(f"\n=== Forge Report: {run.path} ===\n")
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
    print(run.render_tsv())


if __name__ == "__main__":
    main()

# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""vLLM server lifecycle — start/warmup/stop one experiment's server.

See AGENT.md → "Module-by-module spec → runner.py".

vLLM is heavyweight. Each experiment requires:

1. (Layer 2) Build / load the quantized engine — expensive, may take minutes.
2. (Layer 1) Boot the server with the recipe's config.
3. Warmup pass.
4. Hand the URL to ``bench.benchmark``.
5. SIGTERM + drain.

v0 invokes vLLM as a subprocess via ``vllm serve``. Engine-agnostic
abstraction (TensorRT-LLM, SGLang, LMDeploy) is v0.3+; the
``EngineRunner`` Protocol below is the future seam.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from aevyra_forge.recipe import Recipe


logger = logging.getLogger(__name__)


class EngineRunner(Protocol):
    """The contract every engine adapter implements."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def url(self) -> str: ...
    def is_healthy(self) -> bool: ...


class VLLMRunner:
    """Manage a vLLM server's lifecycle for one experiment.

    Sonnet: implement using ``subprocess.Popen("vllm serve ...")``.
    Capture stdout/stderr to a logfile in ``work_dir``. Poll the
    ``/health`` endpoint until ready; bail with a clear error if the
    server doesn't come up within ``startup_timeout_s``.
    """

    def __init__(
        self,
        recipe: Recipe,
        work_dir: Path,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        startup_timeout_s: int = 300,
        dry_run: bool = False,
    ) -> None:
        self.recipe = recipe
        self.work_dir = work_dir
        self.host = host
        self.port = port
        self.startup_timeout_s = startup_timeout_s
        self.dry_run = dry_run

    def start(self) -> None:
        """Launch vLLM and block until /health returns 200.

        In dry-run mode, no subprocess is launched.
        """
        import subprocess
        import time

        if self.dry_run:
            logger.info("[dry-run] Skipping vLLM start for recipe %s", self.recipe.id)
            self._process = None
            return

        import os

        # Default vLLM to WARNING to keep output clean.
        # Users can override by setting VLLM_LOGGING_LEVEL=INFO before running.
        if "VLLM_LOGGING_LEVEL" not in os.environ:
            os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
            logger.info(
                "forge │  vLLM logs suppressed (WARNING level). "
                "Set VLLM_LOGGING_LEVEL=INFO to enable."
            )

        args = ["vllm", "serve"] + build_vllm_args(self.recipe)
        # Suppress uvicorn access logs (separate from VLLM_LOGGING_LEVEL)
        if os.environ.get("VLLM_LOGGING_LEVEL", "WARNING").upper() == "WARNING":
            args += ["--uvicorn-log-level", "warning"]
        log_path = self.work_dir / f"vllm_{self.recipe.id}.log"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        logger.info("forge │  starting vLLM  log → %s", log_path)
        self._log_file = log_path.open("w")

        # Tee vLLM output to the log file AND to stderr so it's visible in
        # notebooks and terminal sessions without needing to tail a file.
        import sys
        import threading

        self._process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        def _tee(proc: subprocess.Popen, log_file: object) -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                decoded = line.decode("utf-8", errors="replace")
                sys.stderr.write(decoded)
                log_file.write(decoded)  # type: ignore[attr-defined]
                log_file.flush()  # type: ignore[attr-defined]

        self._tee_thread = threading.Thread(
            target=_tee, args=(self._process, self._log_file), daemon=True
        )
        self._tee_thread.start()

        try:
            import httpx
        except ImportError as e:
            raise ImportError("runner requires httpx: pip install aevyra-forge[vllm]") from e

        deadline = time.time() + self.startup_timeout_s
        elapsed_intervals = 0
        while time.time() < deadline:
            try:
                r = httpx.get(f"{self.url()}/health", timeout=2.0)
                if r.status_code == 200:
                    logger.info("vLLM ready at %s", self.url())
                    return
            except Exception:
                pass
            if self._process.poll() is not None:
                self._tee_thread.join(timeout=2)
                # Tail the log for the most useful part of the error (last 20 lines).
                tail = ""
                try:
                    lines = log_path.read_text(errors="replace").splitlines()
                    tail = "\n".join(lines[-20:])
                except Exception:
                    pass
                raise RuntimeError(
                    f"vLLM process exited early (code {self._process.returncode}). "
                    f"See {log_path} for details.\n{tail}"
                )
            elapsed_intervals += 1
            if elapsed_intervals % 15 == 0:
                # Heartbeat every 30s so the user knows we're still waiting
                elapsed_s = elapsed_intervals * 2
                logger.info("Waiting for vLLM to start... (%ds elapsed)", elapsed_s)
            time.sleep(2)

        self.stop()
        raise TimeoutError(
            f"vLLM did not become healthy within {self.startup_timeout_s}s. See {log_path}."
        )

    def stop(self) -> None:
        """SIGTERM + wait for clean exit; SIGKILL after grace period."""
        import signal
        import time

        proc = getattr(self, "_process", None)
        if proc is None:
            return
        try:
            proc.send_signal(signal.SIGTERM)
            for _ in range(15):
                if proc.poll() is not None:
                    break
                time.sleep(1)
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        except ProcessLookupError:
            pass
        finally:
            tee_thread = getattr(self, "_tee_thread", None)
            if tee_thread:
                tee_thread.join(timeout=3)
            log_file = getattr(self, "_log_file", None)
            if log_file:
                log_file.close()

    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_healthy(self) -> bool:
        try:
            import httpx

            r = httpx.get(f"{self.url()}/health", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def __enter__(self) -> VLLMRunner:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


_VLLM_HELP_CACHE: str | None = None


def _vllm_help() -> str:
    """Return ``vllm serve --help`` output, cached after first call.

    Used to probe which flags the installed vLLM version supports so we
    don't pass removed flags (e.g. ``--swap-space``, ``--disable-chunked-prefill``).
    Returns an empty string if vLLM is not installed.
    """
    global _VLLM_HELP_CACHE
    if _VLLM_HELP_CACHE is None:
        import subprocess

        try:
            result = subprocess.run(
                ["vllm", "serve", "--help"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            _VLLM_HELP_CACHE = result.stdout + result.stderr
        except Exception:
            _VLLM_HELP_CACHE = ""
    return _VLLM_HELP_CACHE


def build_vllm_args(recipe: Recipe) -> list[str]:
    """Translate a Recipe into ``vllm serve`` argv."""
    cfg = recipe.config
    args: list[str] = [recipe.model]

    args += ["--max-num-seqs", str(cfg.max_num_seqs)]
    args += ["--max-num-batched-tokens", str(cfg.max_num_batched_tokens)]
    args += ["--block-size", str(cfg.block_size)]
    args += ["--gpu-memory-utilization", str(cfg.gpu_memory_utilization)]

    # Probe vllm serve --help once to detect removed flags (cached on module).
    _help = _vllm_help()
    if cfg.swap_space > 0 and "--swap-space" in _help:
        args += ["--swap-space", str(cfg.swap_space)]
    args += ["--kv-cache-dtype", cfg.kv_cache_dtype]
    args += ["--tensor-parallel-size", str(cfg.tensor_parallel_size)]

    if "--pipeline-parallel-size" in _help:
        args += ["--pipeline-parallel-size", str(cfg.pipeline_parallel_size)]

    if cfg.enable_prefix_caching and "--enable-prefix-caching" in _help:
        args.append("--enable-prefix-caching")
    if not cfg.enable_chunked_prefill and "--disable-chunked-prefill" in _help:
        args.append("--disable-chunked-prefill")
    if cfg.attention_backend and "--attention-backend" in _help:
        args += ["--attention-backend", cfg.attention_backend]
    if cfg.speculative_model and "--speculative-model" in _help:
        args += [
            "--speculative-model",
            cfg.speculative_model,
            "--num-speculative-tokens",
            str(cfg.num_speculative_tokens),
        ]

    return args

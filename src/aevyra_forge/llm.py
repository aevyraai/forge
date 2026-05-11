# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LLM client — same contract as ``aevyra_origin.llm``.

See AGENT.md → "Module-by-module spec → llm.py".

The contract is intentionally minimal: ``Callable[[str], str]``. Any
function that fits is a valid agent backend.

Eventually this module + Origin's ``llm.py`` should be extracted into
a shared ``aevyra-common`` package. For v0, copy the implementation
verbatim from Origin (don't reinvent).
"""

from __future__ import annotations

import os
from typing import Callable

LLMFn = Callable[[str], str]


class _AnthropicLLM:
    def __init__(self, client: object, model: str, max_tokens: int, temperature: float) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self.tokens_used: int = 0

    def __call__(self, prompt: str) -> str:
        resp = self._client.messages.create(  # type: ignore[attr-defined]
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.tokens_used += getattr(usage, "input_tokens", 0) + getattr(
                usage, "output_tokens", 0
            )
        return resp.content[0].text


class _OpenAILLM:
    def __init__(self, client: object, model: str, max_tokens: int, temperature: float) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self.tokens_used: int = 0

    def __call__(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(  # type: ignore[attr-defined]
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.tokens_used += getattr(usage, "prompt_tokens", 0) + getattr(
                usage, "completion_tokens", 0
            )
        msg = resp.choices[0].message
        content = msg.content or ""
        if not content:
            content = getattr(msg, "reasoning_content", None) or ""
        return content


def anthropic_llm(
    model: str = "claude-sonnet-4-5",
    *,
    api_key: str | None = None,
    max_tokens: int = 16384,
    temperature: float = 0.0,
) -> _AnthropicLLM:
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise ImportError(
            "anthropic_llm requires the anthropic package. "
            "Install with: pip install aevyra-forge[anthropic]"
        ) from e
    client = Anthropic(api_key=api_key) if api_key else Anthropic()
    return _AnthropicLLM(client, model, max_tokens, temperature)


def openai_llm(
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 16384,
    temperature: float = 0.0,
) -> _OpenAILLM:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "openai_llm requires the openai package. "
            "Install with: pip install aevyra-forge[openai]"
        ) from e
    kwargs: dict = {}
    if api_key is not None:
        kwargs["api_key"] = api_key
    if base_url is not None:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    return _OpenAILLM(client, model, max_tokens, temperature)


def resolve_llm(model_str: str) -> LLMFn:
    """Resolve a ``provider/model`` string into an LLMFn.

    Examples::

        resolve_llm("anthropic/claude-sonnet-4-5")
        resolve_llm("openrouter/qwen/qwen3-8b")
        resolve_llm("openai/gpt-4o")
        resolve_llm("ollama/qwen3:8b")
    """
    parts = model_str.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"model_str must be 'provider/model', got: {model_str!r}")
    provider, model = parts[0].lower(), parts[1]

    if provider == "anthropic":
        return anthropic_llm(model=model)
    if provider == "openai":
        return openai_llm(model=model)
    if provider == "openrouter":
        return openai_llm(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
    if provider == "ollama":
        return openai_llm(model=model, base_url="http://localhost:11434/v1", api_key="ollama")
    return openai_llm(model=model)


__all__ = ["LLMFn", "anthropic_llm", "openai_llm", "resolve_llm"]

# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LLM backends for Forge's decision agent.

The contract is intentionally minimal: ``Callable[[str], str]``.  Any
callable that accepts a prompt string and returns a completion string is
a valid agent backend.

Convenience factories
---------------------

- :func:`anthropic_llm` — Claude via the Anthropic SDK
- :func:`openai_llm`    — any OpenAI-compatible endpoint
- :func:`resolve_llm`   — dispatch from a ``"provider/model"`` string

Supported providers via :func:`resolve_llm`::

    anthropic   anthropic/claude-sonnet-4-6          ANTHROPIC_API_KEY
    openai      openai/gpt-4o                         OPENAI_API_KEY
    openrouter  openrouter/qwen/qwen3-8b              OPENROUTER_API_KEY
    groq        groq/llama-3.3-70b-versatile          GROQ_API_KEY
    together    together/meta-llama/Llama-3.3-70B-... TOGETHER_API_KEY
    fireworks   fireworks/accounts/fireworks/models/… FIREWORKS_API_KEY
    deepinfra   deepinfra/meta-llama/Llama-3-8B       DEEPINFRA_API_KEY
    mistral     mistral/mistral-small-latest          MISTRAL_API_KEY
    ollama      ollama/qwen3:8b                       (local, no key)
    lmstudio    lmstudio/local-model                  (local, no key)

All factories expose a ``tokens_used: int`` attribute that accumulates
across calls, used by the orchestrator's dollar-budget gate.

Eventually this module and ``aevyra_origin.llm`` should be extracted
into a shared ``aevyra-common`` package.
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
            "openai_llm requires the openai package. Install with: pip install aevyra-forge[openai]"
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

    Supported providers
    -------------------

    +--------------+----------------------------------+---------------------------+
    | Provider     | Example string                   | Env var                   |
    +==============+==================================+===========================+
    | anthropic    | anthropic/claude-sonnet-4-6      | ANTHROPIC_API_KEY         |
    | openai       | openai/gpt-4o                    | OPENAI_API_KEY            |
    | openrouter   | openrouter/qwen/qwen3-8b         | OPENROUTER_API_KEY        |
    | groq         | groq/llama-3.3-70b-versatile     | GROQ_API_KEY              |
    | together     | together/meta-llama/Llama-3-8B   | TOGETHER_API_KEY          |
    | fireworks    | fireworks/accounts/fireworks/... | FIREWORKS_API_KEY         |
    | deepinfra    | deepinfra/meta-llama/Llama-3-8B  | DEEPINFRA_API_KEY         |
    | mistral      | mistral/mistral-small-latest     | MISTRAL_API_KEY           |
    | ollama       | ollama/qwen3:8b                  | (none, local)             |
    | lmstudio     | lmstudio/local-model             | (none, local)             |
    +--------------+----------------------------------+---------------------------+

    Any other prefix is forwarded to ``openai_llm`` as-is (useful for
    custom OpenAI-compatible deployments).

    Examples::

        resolve_llm("anthropic/claude-sonnet-4-6")
        resolve_llm("openrouter/qwen/qwen3-8b")
        resolve_llm("groq/llama-3.3-70b-versatile")
        resolve_llm("together/meta-llama/Llama-3.3-70B-Instruct-Turbo")
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

    if provider == "groq":
        return openai_llm(
            model=model,
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY"),
        )

    if provider == "together":
        return openai_llm(
            model=model,
            base_url="https://api.together.xyz/v1",
            api_key=os.environ.get("TOGETHER_API_KEY"),
        )

    if provider == "fireworks":
        return openai_llm(
            model=model,
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.environ.get("FIREWORKS_API_KEY"),
        )

    if provider == "deepinfra":
        return openai_llm(
            model=model,
            base_url="https://api.deepinfra.com/v1/openai",
            api_key=os.environ.get("DEEPINFRA_API_KEY"),
        )

    if provider == "mistral":
        return openai_llm(
            model=model,
            base_url="https://api.mistral.ai/v1",
            api_key=os.environ.get("MISTRAL_API_KEY"),
        )

    if provider == "ollama":
        return openai_llm(
            model=model,
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # ollama ignores the key but openai SDK requires one
        )

    if provider == "lmstudio":
        return openai_llm(
            model=model,
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
        )

    # Unknown provider — try as a plain OpenAI-compatible model string
    return openai_llm(model=model_str)


__all__ = ["LLMFn", "anthropic_llm", "openai_llm", "resolve_llm"]

"""OpenRouter `Pi` factory: OpenAI-compatible endpoint, one pi per model."""

from __future__ import annotations

import os
from typing import Dict, Optional

from ..datagen.samplers import Pi
from ..datagen.utils import make_openai_pi

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter's unified reasoning parameter (https://openrouter.ai/docs).
# Every level is sent as {"reasoning": {"effort": ...}}; "none" is the effort
# level that disables reasoning tokens (e.g. on openai/gpt-5.6-sol).
# "default" -> omit the parameter entirely (provider default).
REASONING_CHOICES = ("default", "none", "minimal", "low", "medium", "high")


def reasoning_extra_body(reasoning: str) -> Optional[dict]:
    """Map a --reasoning flag value to the OpenRouter ``extra_body`` payload."""
    if reasoning not in REASONING_CHOICES:
        raise ValueError(f"reasoning must be one of {REASONING_CHOICES}, got {reasoning!r}")
    if reasoning == "default":
        return None
    return {"reasoning": {"effort": reasoning}}


def make_openrouter_pi(
    model: str,
    temperature: float = 0.7,
    max_output_tokens: int = 256,
    max_retries: int = 4,
    api_key: str | None = None,
    reasoning: str = "default",
) -> Pi:
    """`Pi` via OpenRouter's OpenAI-compatible endpoint; key from ``OPENROUTER_API_KEY``."""
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if api_key is None:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Export it or pass api_key=... "
                "to make_openrouter_pi."
            )
    return make_openai_pi(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_retries=max_retries,
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        extra_body=reasoning_extra_body(reasoning),
    )


def make_openrouter_pis(
    models: list[str],
    temperature: float = 0.7,
    max_output_tokens: int = 256,
    max_retries: int = 4,
    api_key: str | None = None,
    reasoning: str = "default",
) -> Dict[str, Pi]:
    """One `Pi` per unique model name, keyed by model name."""
    return {
        m: make_openrouter_pi(
            model=m,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            api_key=api_key,
            reasoning=reasoning,
        )
        for m in dict.fromkeys(models)
    }

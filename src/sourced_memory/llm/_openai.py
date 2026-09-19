"""OpenAI LLM provider.

Uses the ``responses`` API with a JSON-schema-typed ``response_format`` to
enforce structured output. Works against any OpenAI-compatible endpoint that
supports strict JSON schemas (OpenAI, Azure OpenAI, and most gateway proxies).
"""
from __future__ import annotations
import json
from typing import Any, Mapping

try:
    from openai import OpenAI  # type: ignore
except ImportError as e:  # pragma: no cover - lazy import; extras missing
    raise ImportError(
        "OpenAILLM requires the 'openai' package. "
        "Install with: pip install 'sourced-memory[openai]'"
    ) from e

from .base import LLM, LLMResponseError


class OpenAILLM(LLM):
    """Provider adapter around ``openai.OpenAI``.

    Parameters
    ----------
    model:
        OpenAI model id (e.g. ``"gpt-4o"``).
    client:
        Optional pre-configured ``OpenAI()`` client (for Azure or a proxy).
    temperature:
        Optional temperature. Default 0.0 for deterministic routing.
    """

    def __init__(
        self,
        model: str,
        *,
        client: Any = None,
        temperature: float = 0.0,
    ):
        self._client = client if client is not None else OpenAI()
        self._model = model
        self._temperature = temperature

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        response_schema: Mapping[str, Any],
        response_name: str = "route_result",
    ) -> Mapping[str, Any]:
        # OpenAI structured output uses a wrapping envelope with "name" and "schema".
        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": response_name,
                "schema": dict(response_schema),
                "strict": True,
            },
        }
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=rf,
        )
        text = resp.choices[0].message.content
        if not text:
            raise LLMResponseError(
                f"OpenAI model {self._model!r} returned empty content"
            )
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMResponseError(
                f"OpenAI model {self._model!r} returned non-JSON content: {text[:200]}"
            ) from e

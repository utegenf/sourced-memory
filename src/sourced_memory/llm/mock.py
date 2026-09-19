"""Deterministic in-memory LLM stub for tests and offline demos.

``MockLLM`` accepts either a static response mapping or a callable that
inspects the ``(system, user, response_schema)`` triple and returns a dict.
The mock does no schema validation; test code is expected to return values
that match the router's schema.

Example (canned response)::

    from sourced_memory.llm import MockLLM

    mock = MockLLM({"default": {
        "functional_type": "personal_preference",
        "confidence": 0.9,
        "supported": True,
    }})

Example (callable)::

    def classify(system, user, schema, name):
        if "capital of France" in user:
            return {"functional_type": "external_fact", "confidence": 0.95, "supported": True}
        return {"functional_type": "personal_preference", "confidence": 0.9, "supported": True}

    mock = MockLLM(classify)
"""
from __future__ import annotations
from typing import Any, Callable, Mapping, Union

from .base import LLM, LLMResponseError

_Handler = Union[
    Mapping[str, Mapping[str, Any]],
    Callable[[str, str, Mapping[str, Any], str], Mapping[str, Any]],
]


class MockLLM(LLM):
    """Deterministic LLM stub. Zero network, zero cost, fully seedable in tests."""

    def __init__(self, handler: _Handler):
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        response_schema: Mapping[str, Any],
        response_name: str = "route_result",
    ) -> Mapping[str, Any]:
        self.calls.append({
            "system": system,
            "user": user,
            "response_schema": dict(response_schema),
            "response_name": response_name,
        })
        if callable(self._handler):
            return self._handler(system, user, response_schema, response_name)
        # dict handler: exact key match, else "default", else error
        if user in self._handler:
            return self._handler[user]
        if "default" in self._handler:
            return self._handler["default"]
        raise LLMResponseError(
            f"MockLLM has no handler for user prompt {user!r} and no 'default' entry"
        )

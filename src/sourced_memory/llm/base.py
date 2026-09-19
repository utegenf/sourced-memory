"""Protocol contract for LLM providers used by sourced-memory."""
from __future__ import annotations
from typing import Any, Mapping, Protocol


class LLMResponseError(RuntimeError):
    """Raised when an LLM fails to return a valid structured response."""


class LLM(Protocol):
    """Provider-neutral interface for structured LLM completions.

    Implementations translate the given JSON schema into whatever mechanism
    their provider supports (tool use, JSON mode, function calling) and must
    return a mapping that conforms to the schema. Implementations are free to
    retry internally; on unrecoverable failure they should raise
    :class:`LLMResponseError` with an informative message.

    The core sourced-memory package only needs this one method. Providers may
    expose additional methods (e.g. free-text completion) as they see fit.
    """

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        response_schema: Mapping[str, Any],
        response_name: str = "route_result",
    ) -> Mapping[str, Any]:
        """Return a JSON object matching ``response_schema``.

        ``response_schema`` is a JSON-schema-shaped dict with at least
        ``type: "object"``, ``properties``, and ``required``. ``response_name``
        is a human-readable name for the returned shape (used by providers
        that expose it, e.g. tool-use ``name`` on Anthropic).
        """
        ...

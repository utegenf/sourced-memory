"""Anthropic Claude LLM provider (direct API and Bedrock).

The same class handles both ``anthropic.Anthropic()`` (direct API-key path)
and ``anthropic.AnthropicBedrock()`` (AWS Bedrock path); the SDK exposes both.

Structured output is emitted via forced tool use. The tool's input schema is
built from the caller's ``response_schema`` after inlining any ``$defs``/
``$ref`` pointers, because reasoning-tier Claude models (Opus 5, Sonnet 5)
on Bedrock's toolSpec do not reliably follow ``$ref`` indirection and can
return tool_use payloads with missing referenced fields.

The ``temperature`` parameter is passed through to Anthropic when the caller
supplies one; some newer models reject ``temperature < 1.0`` with a
``deprecated for this model`` validation error. Callers that need to be
model-agnostic can leave ``temperature`` unset.
"""
from __future__ import annotations
from typing import Any, Mapping

try:
    import anthropic  # type: ignore
except ImportError as e:  # pragma: no cover - lazy import; extras missing
    raise ImportError(
        "AnthropicLLM requires the 'anthropic' package. "
        "Install with: pip install 'sourced-memory[anthropic]'"
    ) from e

from .base import LLM, LLMResponseError


def _inline_refs(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively resolve JSON-schema ``$ref`` pointers into their ``$defs``
    targets and strip ``$defs``.

    Reasoning-tier Anthropic models via Bedrock toolSpec do not reliably
    follow ``$ref`` indirection; inlining is transparent to older models.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                if name in defs:
                    return resolve({k: v for k, v in defs[name].items() if k != "$defs"})
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    return resolve(dict(schema))


class AnthropicLLM(LLM):
    """Provider adapter around ``anthropic.Anthropic`` or ``anthropic.AnthropicBedrock``.

    Parameters
    ----------
    model:
        The Anthropic model id or Bedrock inference-profile ARN.
    client:
        A pre-configured ``anthropic.Anthropic()`` or ``anthropic.AnthropicBedrock()``
        instance. If ``None``, a default ``anthropic.Anthropic()`` is constructed
        (which uses ``ANTHROPIC_API_KEY`` from the environment).
    max_tokens:
        Response cap. Generous by default so the router does not truncate.
    temperature:
        Optional. If provided it is sent to the model. Omit for reasoning-tier
        models that only accept temperature=1.0.
    """

    def __init__(
        self,
        model: str,
        *,
        client: Any = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ):
        self._client = client if client is not None else anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        response_schema: Mapping[str, Any],
        response_name: str = "route_result",
    ) -> Mapping[str, Any]:
        inlined = _inline_refs(response_schema)
        # Anthropic tool-use expects "input_schema" (their name); the wrapping is
        # the same {type: object, properties, required} shape as JSON Schema.
        tool = {
            "name": response_name,
            "description": "Return the structured routing decision. You MUST call this tool.",
            "input_schema": {
                "type": "object",
                "properties": inlined.get("properties", {}),
                "required": inlined.get("required", []),
            },
        }
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": response_name},
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        resp = self._client.messages.create(**kwargs)
        for block in resp.content:
            # Anthropic SDK: tool_use blocks have .type == "tool_use" and .input dict
            if getattr(block, "type", None) == "tool_use":
                data = block.input
                if isinstance(data, dict):
                    return data
        raise LLMResponseError(
            f"Anthropic model {self._model!r} did not return a tool_use block; "
            f"got: {[getattr(b, 'type', type(b).__name__) for b in resp.content]}"
        )

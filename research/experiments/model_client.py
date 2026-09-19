"""Model client — the single place the experiments talk to an LLM.

Provider-neutral interface (``complete_text`` / ``complete_structured``) so the experiment code
never contains client plumbing. The reference implementation targets Anthropic Claude via the
official ``anthropic`` SDK; to use a different provider, reimplement these two functions.

Models are named, not the transport. Configure via environment variables:
  AGENT_MODEL   agent model (default: a Claude Sonnet model)
  JUDGE_MODEL   cross-family model used only for paraphrase-robust presence checks
  MAX_TOKENS    max output tokens per call (default: 4096)
"""
import ast
import json
import os
import re
from typing import Optional

import anthropic

AGENT_MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-4-5")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4096"))
TEMPERATURE = 0.0  # greedy decoding for reproducible measurement

_client = anthropic.Anthropic()  # reads the standard API key from the environment


def complete_text(system_prompt: str, user_prompt: str, model_id: str = None,
                  seed: Optional[int] = None) -> str:
    """Single-turn text completion."""
    resp = _client.messages.create(
        model=model_id or AGENT_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def complete_structured(system_prompt: str, user_prompt: str, pydantic_model,
                        model_id: str = None, seed: Optional[int] = None):
    """Structured completion: force the model to return an instance of ``pydantic_model``."""
    schema = pydantic_model.model_json_schema()
    input_schema = {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
        # propagate $defs so nested models resolve their $ref
        **({"$defs": schema["$defs"]} if "$defs" in schema else {}),
    }
    resp = _client.messages.create(
        model=model_id or AGENT_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        tools=[{"name": "return_structured_data",
                "description": "Return the strictly formatted result.",
                "input_schema": input_schema}],
        tool_choice={"type": "tool", "name": "return_structured_data"},
    )
    for block in resp.content:
        if block.type == "tool_use":
            data = _coerce_tool_input(block.input, pydantic_model)
            return pydantic_model(**data)
    raise ValueError(f"{model_id or AGENT_MODEL} returned no structured output")


def _coerce_tool_input(data: dict, pydantic_model) -> dict:
    """Robustness net for models that stringify list fields or echo the schema envelope."""
    coerced = dict(data)
    expected = set(pydantic_model.model_fields)
    if not (expected & coerced.keys()) and isinstance(coerced.get("properties"), dict):
        coerced = dict(coerced["properties"])
    for field_name, field_info in pydantic_model.model_fields.items():
        value = coerced.get(field_name)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith(("[", "{")):
                try:
                    coerced[field_name] = json.loads(stripped)
                except (ValueError, TypeError):
                    try:
                        coerced[field_name] = ast.literal_eval(stripped)
                    except (ValueError, SyntaxError):
                        if stripped.startswith("[") and str(field_info.annotation).startswith(("typing.List", "list")):
                            inner = stripped[1:-1].strip()
                            items = re.split(r"'\s*,\s*'", inner)
                            coerced[field_name] = [it.strip().strip("'\"") for it in items if it.strip()]
    return coerced

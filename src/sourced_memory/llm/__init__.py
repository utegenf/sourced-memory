"""Provider-neutral LLM interface for sourced-memory.

The core package ships only ``MockLLM``. Provider integrations
(``AnthropicLLM``, ``OpenAILLM``) are lazily imported from optional extras
so importing this package does not require any external SDK.

Usage::

    from sourced_memory.llm import MockLLM
    from sourced_memory.llm import AnthropicLLM   # requires ``pip install sourced-memory[anthropic]``
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from .base import LLM, LLMResponseError
from .mock import MockLLM

__all__ = ["LLM", "LLMResponseError", "MockLLM", "AnthropicLLM", "OpenAILLM"]


def __getattr__(name):
    if name == "AnthropicLLM":
        try:
            from ._anthropic import AnthropicLLM
        except ImportError as e:
            raise ImportError(
                "AnthropicLLM requires the 'anthropic' SDK. "
                "Install it with: pip install 'sourced-memory[anthropic]'"
            ) from e
        return AnthropicLLM
    if name == "OpenAILLM":
        try:
            from ._openai import OpenAILLM
        except ImportError as e:
            raise ImportError(
                "OpenAILLM requires the 'openai' SDK. "
                "Install it with: pip install 'sourced-memory[openai]'"
            ) from e
        return OpenAILLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from ._anthropic import AnthropicLLM
    from ._openai import OpenAILLM

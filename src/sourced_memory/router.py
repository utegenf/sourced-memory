"""Content-only functional routing.

Routers see the item text and nothing else. Provenance is applied afterward by
the policy; a router implementation cannot access `Source` even by mistake.
"""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
from .models import FunctionalType

if TYPE_CHECKING:
    from .llm.base import LLM

@dataclass(frozen=True)
class RouteResult:
    functional_type: FunctionalType
    confidence: float = 1.0
    supported: bool = True
    summarized_content: str | None = None

class Router(Protocol):
    def route(self, content: str) -> RouteResult: ...

class CallableRouter:
    def __init__(self, classifier: Callable[[str], RouteResult]):
        self._classifier = classifier
    def route(self, content: str) -> RouteResult:
        return self._classifier(content)

class RuleBasedRouter:
    """Dependency-free example router; use an LLM-backed Router for production."""
    def route(self, content: str) -> RouteResult:
        text = content.lower().strip()
        if text.endswith("?") or text.startswith(("please ", "can you ", "could you ", "remind me")):
            return RouteResult(FunctionalType.EVENT, summarized_content=content)
        if any(x in text for x in ("i like ", "i love ", "i prefer ", "i dislike ", "i hate ", "my favorite ")):
            return RouteResult(FunctionalType.PERSONAL_PREFERENCE, summarized_content=content)
        if any(x in text for x in ("my ", "our ", "i live ", "i work ", "my partner", "my manager")):
            return RouteResult(FunctionalType.RELATIONAL_FACT, summarized_content=content)
        if any(x in text for x in ("always ", "never ", "from now on", "please always", "please never")):
            return RouteResult(FunctionalType.GENERAL_RULE, summarized_content=content)
        return RouteResult(FunctionalType.EXTERNAL_FACT, summarized_content=content)


class NullRouter:
    """Passthrough router for callers that already know the functional type.

    Emits a fixed ``functional_type`` and a fixed ``confidence`` for every
    call. Useful when the calling application has its own classifier and
    wants only the source/type admission policy from sourced-memory.
    """

    def __init__(
        self,
        functional_type: FunctionalType = FunctionalType.EXTERNAL_FACT,
        *,
        confidence: float = 1.0,
        supported: bool = True,
    ):
        self._ft = functional_type
        self._confidence = confidence
        self._supported = supported

    def route(self, content: str) -> RouteResult:
        return RouteResult(
            functional_type=self._ft,
            confidence=self._confidence,
            supported=self._supported,
            summarized_content=content,
        )


_DEFAULT_LLM_ROUTER_SYSTEM = (
    "You are a content router for a persistent-memory agent. You classify a "
    "single incoming statement by its FUNCTIONAL TYPE only. You do not see "
    "the source or channel; you do not decide trust; you do not decide "
    "whether the statement should be remembered. You emit only:\n"
    "  functional_type: one of PERSONAL_PREFERENCE, GENERAL_RULE, "
    "RELATIONAL_FACT, EXTERNAL_FACT, EVENT\n"
    "  confidence: 0..1, how confident you are in the classification\n"
    "  supported: true if the statement is a well-formed durable claim "
    "worth admitting for further processing, false if it is an artifact "
    "(empty, meta-instruction, request)\n"
    "  summarized_content: a concise canonical form of the statement\n\n"
    "Category definitions:\n"
    "  PERSONAL_PREFERENCE: a durable preference or taste of the person "
    "(e.g. 'I prefer tea', 'I love hiking on weekends').\n"
    "  GENERAL_RULE: a standing instruction or rule about how to behave "
    "(e.g. 'always use ARIMA as a first baseline', 'never contact me on Fridays').\n"
    "  RELATIONAL_FACT: a stable fact about this person's world "
    "(e.g. 'my manager is Alice', 'I work at ACME').\n"
    "  EXTERNAL_FACT: a fact about the world outside the person "
    "(e.g. 'the capital of France is Paris').\n"
    "  EVENT: a transient one-off occurrence or request "
    "(e.g. 'please remind me to call Alice', 'I ate lunch at 1pm').\n"
)

_LLM_ROUTER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "functional_type": {
            "type": "string",
            "enum": [ft.value for ft in FunctionalType],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "supported": {"type": "boolean"},
        "summarized_content": {"type": "string"},
    },
    "required": ["functional_type", "confidence", "supported"],
}


class LLMRouter:
    """Router backed by an :class:`sourced_memory.llm.LLM` provider.

    Uses forced structured output to classify content into one of the five
    functional types. The default prompt is the paper's generalized gate
    prompt; override ``system_prompt`` to customize.

    Example::

        from sourced_memory.llm import AnthropicLLM
        from sourced_memory.router import LLMRouter

        router = LLMRouter(AnthropicLLM("claude-sonnet-4-5"))
        result = router.route("I love hiking on weekends.")
        # RouteResult(functional_type=PERSONAL_PREFERENCE, confidence=..., ...)
    """

    def __init__(
        self,
        llm: "LLM",
        *,
        system_prompt: str = _DEFAULT_LLM_ROUTER_SYSTEM,
        response_name: str = "route_result",
    ):
        self._llm = llm
        self._system = system_prompt
        self._name = response_name

    def route(self, content: str) -> RouteResult:
        raw = self._llm.complete_json(
            system=self._system,
            user=f"Classify this statement:\n{content}",
            response_schema=_LLM_ROUTER_SCHEMA,
            response_name=self._name,
        )
        return RouteResult(
            functional_type=FunctionalType(raw["functional_type"]),
            confidence=float(raw.get("confidence", 1.0)),
            supported=bool(raw.get("supported", True)),
            summarized_content=raw.get("summarized_content") or content,
        )

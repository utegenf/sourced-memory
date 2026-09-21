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

_PP_FIRST_PERSON = (
    "i like ", "i love ", "i prefer ", "i dislike ", "i hate ",
    "my favorite ",
)
_PP_THIRD_PERSON = (
    # "the user X's Y" preference verbs
    "the user likes",  "the user loves",  "the user prefers",
    "the user dislikes", "the user hates", "the user enjoys",
    "user likes ", "user loves ", "user prefers ",
    "user hates ", "user enjoys ", "user dislikes ",
    # copular "the user is / has been / was ..." claims about the user
    "the user is ", "the user was ", "the user has been ",
    "the user's favorite ",
)

_RF_FIRST_PERSON = (
    "my ", "our ", "i live ", "i work ", "my partner", "my manager",
)
_RF_THIRD_PERSON = (
    "the user's manager", "the user's partner", "the user's team",
    "the user lives ", "the user works ", "the user has ",
    "user's manager", "user's partner",
)


class RuleBasedRouter:
    """Dependency-free keyword-based router.

    Catches obvious first-person phrasing ("I love X") *and* the common
    third-person forms ("the user hates X", "user prefers Y", "the user is
    an expert Z") that appear in real memory-poisoning attempts embedded
    in RAG results, tool outputs, or sub-agent replies. Still keyword-only:
    it will miss adversarial phrasings that avoid these patterns. For
    paper-quality routing use :class:`LLMRouter`. This router exists so
    the library ships with a router that is honest about the "no LLM
    available" case, not so it matches the paper's validated coverage.
    """
    def route(self, content: str) -> RouteResult:
        text = content.lower().strip()
        if text.endswith("?") or text.startswith(("please ", "can you ", "could you ", "remind me")):
            return RouteResult(FunctionalType.EVENT, summarized_content=content)
        # PERSONAL_PREFERENCE checked before RELATIONAL_FACT: "the user's
        # favorite color" is a preference, not a relational fact.
        if any(x in text for x in _PP_FIRST_PERSON) or any(x in text for x in _PP_THIRD_PERSON):
            return RouteResult(FunctionalType.PERSONAL_PREFERENCE, summarized_content=content)
        if any(x in text for x in _RF_FIRST_PERSON) or any(x in text for x in _RF_THIRD_PERSON):
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

    **Caching threat model.** LLMRouter can cache classifications keyed on
    a hash of the content string. Cache design invariant: **the router
    never sees the source and the cache never keys on source.** Two
    observations with identical content and different sources SHARE the
    cached classification, and both are correctly source-gated afterward
    by the downstream policy layer (which does not consult the cache).
    This is the paper's design (content and origin are separated at
    admission), not an accident of the cache implementation. See tests
    ``test_llm_router_cache_key_is_content_only`` and
    ``test_source_gating_never_uses_router_cache`` for the invariants
    made structural.

    Example::

        from sourced_memory.llm import AnthropicLLM
        from sourced_memory.router import LLMRouter

        router = LLMRouter(AnthropicLLM("claude-sonnet-4-5"))
        result = router.route("I love hiking on weekends.")
    """

    def __init__(
        self,
        llm: "LLM",
        *,
        system_prompt: str = _DEFAULT_LLM_ROUTER_SYSTEM,
        response_name: str = "route_result",
        cache_size: int = 1024,
    ):
        self._llm = llm
        self._system = system_prompt
        self._name = response_name
        # Content-hash keyed cache. Only content participates in the key,
        # by construction. If cache_size is <= 0 the cache is disabled and
        # every call reaches the LLM (useful for cost-tracking tests).
        self._cache_size = cache_size
        self._cache: dict[str, RouteResult] = {}
        # Preserve FIFO for a bounded cache without adding a dep.
        self._cache_order: list[str] = []

    def route(self, content: str) -> RouteResult:
        if self._cache_size > 0:
            key = content   # exact string is the key; hashing adds no value here
            hit = self._cache.get(key)
            if hit is not None:
                return hit
        raw = self._llm.complete_json(
            system=self._system,
            user=f"Classify this statement:\n{content}",
            response_schema=_LLM_ROUTER_SCHEMA,
            response_name=self._name,
        )
        result = RouteResult(
            functional_type=FunctionalType(raw["functional_type"]),
            confidence=float(raw.get("confidence", 1.0)),
            supported=bool(raw.get("supported", True)),
            summarized_content=raw.get("summarized_content") or content,
        )
        if self._cache_size > 0:
            self._cache[content] = result
            self._cache_order.append(content)
            if len(self._cache_order) > self._cache_size:
                evict = self._cache_order.pop(0)
                self._cache.pop(evict, None)
        return result

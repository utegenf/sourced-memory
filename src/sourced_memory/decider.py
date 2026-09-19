"""Storage-free admission decider.

``Decider`` is a lightweight facade around ``Router`` + ``Policy``. Where
``SourceAwareMemory`` also owns state (belief/candidate/episodic lists),
``Decider`` returns only an :class:`AdmissionRecord` and does no storage.

Adapters that sit in front of an *existing* memory system (Mem0, Zep,
LangMem, application-owned stores) use ``Decider`` and pass the decision
back to that store rather than duplicating storage inside sourced-memory.

Example::

    from sourced_memory import Decider, TrustPolicy
    from sourced_memory.router import LLMRouter
    from sourced_memory.llm import AnthropicLLM

    decider = Decider(
        policy=TrustPolicy.reference(),
        router=LLMRouter(AnthropicLLM("claude-sonnet-4-5")),
    )
    record = decider.decide("I love hiking.", source="user", trusted=True)
    if record.decision is AdmissionDecision.BELIEF:
        my_backend.write(...)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from .models import AdmissionDecision, FunctionalType, Source
from .policy import SourceTypePolicy
from .router import Router, RuleBasedRouter


@dataclass(frozen=True)
class DecisionRecord:
    """The result of a single admission decision. Storage-free."""
    content: str
    source: Source
    decision: AdmissionDecision
    functional_type: FunctionalType
    confidence: float
    supported: bool
    summarized_content: str


class Decider:
    """Admission decision as a pure function of (content, source).

    Composed of a :class:`Router` (content → functional type) and a
    :class:`Policy` (source + type → destination). Returns a
    :class:`DecisionRecord` describing the outcome; does not store anything.
    """

    def __init__(
        self,
        *,
        policy: SourceTypePolicy | None = None,
        router: Router | None = None,
        trusted_sources: set[str] | None = None,
    ):
        # ``policy`` is the (source, type) rule table. ``trusted_sources`` is a
        # shorthand for which source names should carry ``Source.trusted=True``.
        # They are orthogonal and both can be supplied.
        self.policy = policy if policy is not None else SourceTypePolicy.reference()
        self.router = router if router is not None else RuleBasedRouter()
        self._trusted_sources = set(trusted_sources or {"user"})

    def decide(
        self,
        content: str,
        *,
        source: str,
        source_id: str | None = None,
        trusted: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DecisionRecord:
        if not content or not content.strip():
            raise ValueError("content must be non-empty")
        if not source:
            raise ValueError("source must be non-empty")
        if trusted is None:
            trusted = source in self._trusted_sources
        src = Source(
            name=source,
            source_id=source_id,
            trusted=trusted,
            metadata=metadata or {},
        )
        route = self.router.route(content)
        if not route.supported:
            decision = AdmissionDecision.REJECT
        else:
            decision = self.policy.decide(src, route.functional_type)
        return DecisionRecord(
            content=content,
            source=src,
            decision=decision,
            functional_type=route.functional_type,
            confidence=route.confidence,
            supported=route.supported,
            summarized_content=route.summarized_content or content,
        )

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

from .channel import Channel
from .models import AdmissionDecision, FunctionalType, Source
from .policy import TrustPolicy
from .router import Router, RuleBasedRouter


@dataclass(frozen=True)
class DecisionRecord:
    """The result of a single admission decision. Storage-free.

    ``backing_store_id`` is populated by adapters (e.g. :class:`WrappedMem0`)
    with the id the underlying store returned when a BELIEF was written.
    It is what makes ``ProtectedMemory.purge(source_id=...)`` able to
    delete the corresponding record from the backing store on demand.
    None when the underlying store returned no id, or when the decision
    was not BELIEF (nothing was written to the backing store).
    """
    content: str
    source: Source
    decision: AdmissionDecision
    functional_type: FunctionalType
    confidence: float
    supported: bool
    summarized_content: str
    backing_store_id: str | None = None


class Decider:
    """Admission decision as a pure function of (content, source).

    Composed of a :class:`Router` (content → functional type) and a
    :class:`Policy` (source + type → destination). Returns a
    :class:`DecisionRecord` describing the outcome; does not store anything.
    """

    def __init__(
        self,
        *,
        policy: TrustPolicy | None = None,
        router: Router | None = None,
        trusted_sources: set[str] | None = None,
    ):
        # ``policy`` is the (source, type) rule table. ``trusted_sources`` is a
        # shorthand for which source names should carry ``Source.trusted=True``.
        # They are orthogonal and both can be supplied.
        self.policy = policy if policy is not None else TrustPolicy.reference()
        self.router = router if router is not None else RuleBasedRouter()
        self._trusted_sources = set(trusted_sources or {"user"})

    def channel(
        self,
        name: str,
        *,
        trusted: bool | None = None,
        source_id: str | None = None,
    ) -> Channel:
        """Create a :class:`Channel` bound to this Decider.

        If ``trusted`` is omitted, it defaults to whether ``name`` is in the
        Decider's ``trusted_sources``. Prefer explicit ``trusted=`` in new code.
        """
        if trusted is None:
            trusted = name in self._trusted_sources
        return Channel(name=name, trusted=trusted, source_id=source_id, _target=self)

    def decide(
        self,
        content: str,
        *,
        source: "str | Channel",
        source_id: str | None = None,
        trusted: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DecisionRecord:
        if isinstance(source, Channel):
            source_name = source.name
            if trusted is None:
                trusted = source.trusted
            if source_id is None:
                source_id = source.source_id
        else:
            source_name = source
            if trusted is None:
                trusted = source_name in self._trusted_sources
        if not content or not content.strip():
            raise ValueError("content must be non-empty")
        if not source_name:
            raise ValueError("source must be non-empty")
        src = Source(
            name=source_name,
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

    # Aliased method name so a Decider can stand in for any target Channel expects.
    def observe(self, content: str, **kwargs: Any) -> DecisionRecord:
        return self.decide(content, **kwargs)

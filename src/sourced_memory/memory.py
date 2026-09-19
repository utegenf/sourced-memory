"""High-level source-aware memory admission layer."""
from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from .models import AdmissionDecision, Belief, CandidateEvidence, Experience, FunctionalType, Source
from .policy import SourceTypePolicy
from .router import RouteResult, Router, RuleBasedRouter

@dataclass(frozen=True)
class AdmissionRecord:
    experience_id: str
    functional_type: FunctionalType
    decision: AdmissionDecision
    confidence: float
    supported: bool

class SourceAwareMemory:
    """Separates content interpretation, source-conditioned admission, and state."""

    def __init__(self, *, trusted_sources: Iterable[str] | None = None,
                 policy: SourceTypePolicy | None = None, router: Router | None = None):
        if policy is not None and trusted_sources is not None:
            raise ValueError("Pass either policy or trusted_sources, not both.")
        if policy is None:
            policy = SourceTypePolicy.reference()
        self.policy = policy
        self.router = router or RuleBasedRouter()
        self._trusted_sources = set(trusted_sources or {"user"})
        self._experiences: list[Experience] = []
        self._beliefs: list[Belief] = []
        self._candidates: list[CandidateEvidence] = []
        self._episodic: list[Experience] = []
        self._decisions: list[AdmissionRecord] = []

    def observe(self, content: str, *, source: str, source_id: str | None = None,
                trusted: bool | None = None, episode_id: str | None = None,
                metadata: dict[str, Any] | None = None) -> Experience:
        if not content or not content.strip():
            raise ValueError("content must be non-empty")
        if not source:
            raise ValueError("source must be non-empty")
        if trusted is None:
            trusted = source in self._trusted_sources
        experience = Experience(
            content=content,
            source=Source(name=source, source_id=source_id, trusted=trusted),
            episode_id=episode_id,
            metadata=metadata or {},
        )
        self._experiences.append(experience)
        return experience

    def consolidate(self) -> list[AdmissionRecord]:
        processed = {x.experience_id for x in self._decisions}
        for experience in self._experiences:
            if experience.id in processed:
                continue
            route = self.router.route(experience.content)
            decision = (self.policy.decide(experience.source, route.functional_type)
                        if route.supported else AdmissionDecision.REJECT)
            self._apply(experience, route, decision)
            self._decisions.append(AdmissionRecord(
                experience_id=experience.id,
                functional_type=route.functional_type,
                decision=decision,
                confidence=route.confidence,
                supported=route.supported,
            ))
        return list(self._decisions)

    def _apply(self, experience: Experience, route: RouteResult, decision: AdmissionDecision) -> None:
        content = route.summarized_content or experience.content
        if decision is AdmissionDecision.BELIEF:
            self._beliefs.append(Belief(content=content, functional_type=route.functional_type,
                                        source=experience.source, experience_id=experience.id,
                                        confidence=route.confidence, metadata=dict(experience.metadata)))
        elif decision is AdmissionDecision.CANDIDATE:
            self._candidates.append(CandidateEvidence(
                content=content, functional_type=route.functional_type, source=experience.source,
                experience_id=experience.id, confidence=route.confidence,
                metadata=dict(experience.metadata)))
        elif decision is AdmissionDecision.EPISODIC:
            self._episodic.append(experience)

    def beliefs(self) -> list[Belief]:
        """Return currently admitted beliefs. Call consolidate() to advance state."""
        return list(self._beliefs)

    def candidates(self) -> list[CandidateEvidence]:
        """Return currently held candidate evidence. Call consolidate() to advance state."""
        return list(self._candidates)

    def episodic(self) -> list[Experience]:
        """Return episodic memories. Call consolidate() to advance state."""
        return list(self._episodic)

    def decisions(self) -> list[AdmissionRecord]:
        """Return recorded admission decisions. Call consolidate() to advance state."""
        return list(self._decisions)

    def pending(self) -> int:
        """Number of buffered experiences that have not yet been consolidated."""
        processed = {x.experience_id for x in self._decisions}
        return sum(1 for x in self._experiences if x.id not in processed)

    def clear(self) -> None:
        self._experiences.clear()
        self._beliefs.clear()
        self._candidates.clear()
        self._episodic.clear()
        self._decisions.clear()

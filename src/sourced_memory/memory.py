"""High-level source-aware memory admission layer."""
from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from .channel import Channel
from .models import AdmissionDecision, Belief, CandidateEvidence, Experience, FunctionalType, Source
from .policy import TrustPolicy
from .router import RouteResult, Router, RuleBasedRouter


@dataclass(frozen=True)
class AdmissionRecord:
    experience_id: str
    functional_type: FunctionalType
    decision: AdmissionDecision
    confidence: float
    supported: bool


class SourceAwareMemory:
    """Separates content interpretation, source-conditioned admission, and state.

    The preferred entry point for admission is :meth:`channel`, which returns
    a :class:`Channel` bound to this memory. Configure channels once at
    setup, then call ``channel.observe(...)``; the source name, trust flag,
    and source_id are supplied by the channel automatically.

    The lower-level :meth:`observe` remains available for cases where the
    channel is chosen at runtime.
    """

    def __init__(
        self,
        *,
        trusted_sources: Iterable[str] | None = None,
        policy: TrustPolicy | None = None,
        router: Router | None = None,
    ):
        # trusted_sources and policy are orthogonal: policy is the (source,
        # type) rule table; trusted_sources is a shorthand for which source
        # names should carry ``Source.trusted=True`` by default. Both may be
        # supplied together.
        if policy is None:
            policy = TrustPolicy.reference()
        self.policy = policy
        self.router = router or RuleBasedRouter()
        self._trusted_sources = set(trusted_sources or {"user"})
        self._experiences: list[Experience] = []
        self._beliefs: list[Belief] = []
        self._candidates: list[CandidateEvidence] = []
        self._episodic: list[Experience] = []
        self._decisions: list[AdmissionRecord] = []

    # ------------------------------------------------------------------
    # Channel API (preferred): bind a source at setup time, reuse it.
    # ------------------------------------------------------------------

    def channel(
        self,
        name: str,
        *,
        trusted: bool | None = None,
        source_id: str | None = None,
    ) -> Channel:
        """Create a :class:`Channel` bound to this memory.

        If ``trusted`` is omitted, it defaults to whether ``name`` appears
        in the memory's ``trusted_sources`` set. Prefer explicit
        ``trusted=`` in new code.
        """
        if trusted is None:
            trusted = name in self._trusted_sources
        return Channel(name=name, trusted=trusted, source_id=source_id, _target=self)

    def observe(
        self,
        content: str,
        *,
        source: "str | Channel",
        source_id: str | None = None,
        trusted: bool | None = None,
        episode_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Experience:
        """Escape hatch: observe with an explicit source, string or Channel.

        Prefer ``memory.channel(...).observe(...)`` in application code;
        this form exists for cases where the channel is chosen at runtime.
        """
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
        experience = Experience(
            content=content,
            source=Source(name=source_name, source_id=source_id, trusted=trusted),
            episode_id=episode_id,
            metadata=metadata or {},
        )
        self._experiences.append(experience)
        return experience

    # ------------------------------------------------------------------
    # Consolidation and state advance
    # ------------------------------------------------------------------

    def consolidate(self) -> list[AdmissionRecord]:
        processed = {x.experience_id for x in self._decisions}
        for experience in self._experiences:
            if experience.id in processed:
                continue
            route = self.router.route(experience.content)
            decision = (
                self.policy.decide(experience.source, route.functional_type)
                if route.supported
                else AdmissionDecision.REJECT
            )
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
            self._beliefs.append(Belief(
                content=content, functional_type=route.functional_type,
                source=experience.source, experience_id=experience.id,
                confidence=route.confidence, metadata=dict(experience.metadata),
            ))
        elif decision is AdmissionDecision.CANDIDATE:
            self._candidates.append(CandidateEvidence(
                content=content, functional_type=route.functional_type,
                source=experience.source, experience_id=experience.id,
                confidence=route.confidence, metadata=dict(experience.metadata),
            ))
        elif decision is AdmissionDecision.EPISODIC:
            self._episodic.append(experience)

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Remediation
    # ------------------------------------------------------------------

    def purge(self, *, source_id: str) -> int:
        """Remove all state tagged with ``source_id``. Returns the number of
        records removed across beliefs, candidates, episodic, experiences,
        and decisions. Deterministic; no LLM in the loop.

        Use this to revoke a compromised session or roll back memory
        written under a specific source identity. Compose with
        ``channel.session(source_id)`` at write time so the tag exists on
        every affected record.
        """
        # Find experience_ids to be purged before removing them, so we can
        # also drop decisions that reference them.
        exp_ids_to_drop = {
            e.id for e in self._experiences if e.source.source_id == source_id
        }
        exp_ids_to_drop.update(b.experience_id for b in self._beliefs
                               if b.source.source_id == source_id)
        exp_ids_to_drop.update(c.experience_id for c in self._candidates
                               if c.source.source_id == source_id)

        removed = 0
        n = len(self._beliefs)
        self._beliefs = [b for b in self._beliefs if b.source.source_id != source_id]
        removed += n - len(self._beliefs)

        n = len(self._candidates)
        self._candidates = [c for c in self._candidates if c.source.source_id != source_id]
        removed += n - len(self._candidates)

        n = len(self._episodic)
        self._episodic = [e for e in self._episodic if e.source.source_id != source_id]
        removed += n - len(self._episodic)

        n = len(self._experiences)
        self._experiences = [e for e in self._experiences if e.source.source_id != source_id]
        removed += n - len(self._experiences)

        n = len(self._decisions)
        self._decisions = [d for d in self._decisions if d.experience_id not in exp_ids_to_drop]
        removed += n - len(self._decisions)

        return removed

    def clear(self) -> None:
        self._experiences.clear()
        self._beliefs.clear()
        self._candidates.clear()
        self._episodic.clear()
        self._decisions.clear()

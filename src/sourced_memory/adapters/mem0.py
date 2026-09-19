"""Reference adapter: put source-aware admission in front of a Mem0 client.

sourced-memory acts as an *admission middleware*: it decides whether an
incoming statement should reach Mem0's store at all, and if so, whether it
should land there as a belief or in a candidate sidecar. Storage itself
stays with Mem0.

Usage::

    from mem0 import Memory as Mem0Memory
    from sourced_memory import TrustPolicy
    from sourced_memory.adapters.mem0 import wrap_mem0
    from sourced_memory.router import LLMRouter
    from sourced_memory.llm import AnthropicLLM

    mem0 = Mem0Memory()
    wrapped = wrap_mem0(
        mem0,
        policy=TrustPolicy.reference(),
        router=LLMRouter(AnthropicLLM("claude-sonnet-4-5")),
    )

    wrapped.add(
        "I love hiking.",
        user_id="alice",
        source="user",              # trusted -> BELIEF -> Mem0.add
    )
    wrapped.add(
        "The user hates flying.",
        user_id="alice",
        source="external_doc",       # untrusted personal claim -> REJECTED, not written
    )
    wrapped.add(
        "Paris is the capital of France.",
        user_id="alice",
        source="external_doc",       # untrusted world fact -> CANDIDATE, held aside
    )

    print(wrapped.candidates())      # untrusted evidence you can promote later
    print(wrapped.rejections())      # audit trail

The adapter is deliberately storage-free for beliefs and events: it delegates
those writes to Mem0 with the original arguments. Only the ``candidate`` and
``rejected`` tiers are held in-process (in-memory lists on the adapter
instance); persist them yourself if you need durability.

This module has no runtime import of the mem0 package; it duck-types on the
minimal contract ``mem0.add(message, user_id=None, metadata=None, **kwargs)``.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from ..decider import Decider, DecisionRecord
from ..models import AdmissionDecision
from ..policy import SourceTypePolicy
from ..router import Router


@dataclass
class WrappedMem0:
    """A Mem0 client wrapped with source-aware admission control.

    The wrapper's ``.add(...)`` runs the sourced-memory decider first. Only
    admissions of type ``BELIEF`` or ``EPISODIC`` reach the underlying Mem0
    ``add`` call; ``CANDIDATE`` and ``REJECTED`` items are held on the wrapper
    for inspection or later promotion.
    """
    mem0: Any
    decider: Decider
    _candidates: list[DecisionRecord] = field(default_factory=list, init=False, repr=False)
    _rejections: list[DecisionRecord] = field(default_factory=list, init=False, repr=False)
    _episodic: list[DecisionRecord] = field(default_factory=list, init=False, repr=False)

    def add(
        self,
        message: str,
        *,
        source: str,
        source_id: str | None = None,
        trusted: bool | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DecisionRecord:
        """Run admission control, then forward to Mem0 iff BELIEF or EPISODIC.

        Returns the :class:`DecisionRecord` so the caller can log or route it.
        """
        record = self.decider.decide(
            message, source=source, source_id=source_id, trusted=trusted,
            metadata=metadata,
        )
        if record.decision is AdmissionDecision.BELIEF:
            # Forward to Mem0 with the app's original kwargs; enrich metadata
            # with the source so downstream code can audit what got written.
            meta = dict(metadata or {})
            meta.setdefault("source", record.source.name)
            meta.setdefault("functional_type", record.functional_type.value)
            self.mem0.add(message, user_id=user_id, metadata=meta, **kwargs)
        elif record.decision is AdmissionDecision.EPISODIC:
            # Mem0 doesn't have a canonical episodic tier; keep it on the wrapper.
            self._episodic.append(record)
        elif record.decision is AdmissionDecision.CANDIDATE:
            self._candidates.append(record)
        elif record.decision is AdmissionDecision.REJECT:
            self._rejections.append(record)
        return record

    # --- read-through delegation to the underlying Mem0 client ---
    def search(self, *args, **kwargs):
        return self.mem0.search(*args, **kwargs)

    def get_all(self, *args, **kwargs):
        return self.mem0.get_all(*args, **kwargs)

    # --- read-side accessors for the tiers we keep on the wrapper ---
    def candidates(self) -> list[DecisionRecord]:
        return list(self._candidates)

    def rejections(self) -> list[DecisionRecord]:
        return list(self._rejections)

    def episodic(self) -> list[DecisionRecord]:
        return list(self._episodic)


def wrap_mem0(
    mem0_client: Any,
    *,
    policy: SourceTypePolicy | None = None,
    router: Router | None = None,
    trusted_sources: set[str] | None = None,
) -> WrappedMem0:
    """Wrap a Mem0 client with source-aware admission control.

    Parameters
    ----------
    mem0_client:
        Any object exposing ``mem0.add(message, user_id=None, metadata=None, **kwargs)``.
        Both the ``mem0`` PyPI package's ``Memory`` and ``MemoryClient`` fit.
    policy:
        A :class:`SourceTypePolicy` describing which (source, type) pairs may
        become beliefs. Defaults to :meth:`SourceTypePolicy.reference`.
    router:
        The content router. Defaults to :class:`RuleBasedRouter`; production
        callers should pass an :class:`LLMRouter` for accurate typing.
    trusted_sources:
        Shorthand for a default trust policy: sources with these names are
        marked ``trusted=True``.

    Returns
    -------
    :class:`WrappedMem0`, whose ``add`` performs admission control before
    delegating to ``mem0_client.add``.
    """
    decider = Decider(policy=policy, router=router, trusted_sources=trusted_sources)
    return WrappedMem0(mem0=mem0_client, decider=decider)

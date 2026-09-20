"""High-level product API: ``protect(store, trusted=[...], untrusted=[...])``.

The rest of this package is the mechanism. ``protect()`` is the mechanism
turned into a five-minute developer experience.

Usage
-----

Ephemeral, in-process (great for tests, demos, and the README quickstart)::

    from sourced_memory import protect

    memory = protect(trusted=["user"], untrusted=["web", "tool"])
    memory.user.add("I love hiking.")
    memory.web.add("The user hates hiking and prefers gaming.")

    for entry in memory.audit():
        print(entry)
    # ✓ BELIEF   "I love hiking."                             (user)
    # ✗ REJECT   "The user hates hiking and prefers gaming."  (web, personal_preference from untrusted)

In front of an existing memory system (Mem0, Zep, application-owned store)::

    from mem0 import Memory as Mem0Memory
    memory = protect(Mem0Memory(), trusted=["user"], untrusted=["web", "tool"])

Design boundaries
-----------------
- Undeclared channels raise :class:`UnknownChannelError`. Loud beats silent
  for a security library; the caller must declare every source it will use.
- The facade hides the two-phase (observe / consolidate) split of the raw
  :class:`SourceAwareMemory`. ``.audit()``, ``.beliefs()``, ``.candidates()``,
  ``.episodic()``, ``.purge()`` do the right thing regardless of backing store.
- The router and the policy are still swappable. Pass ``router=`` or
  ``policy=`` to override the defaults (``RuleBasedRouter`` / paper's
  reference policy).
- Advanced use cases (custom adapters, runtime-dynamic channels, direct
  access to Decider) still work: import from ``sourced_memory.advanced``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .adapters.mem0 import WrappedMem0, wrap_mem0
from .decider import DecisionRecord
from .memory import SourceAwareMemory
from .models import AdmissionDecision, FunctionalType, Source
from .policy import TrustPolicy
from .router import Router, RuleBasedRouter


class UnknownChannelError(RuntimeError):
    """Raised when an undeclared channel is accessed on a ProtectedMemory."""


_DECISION_MARKS = {
    AdmissionDecision.BELIEF:    "✓",   # heavy check mark
    AdmissionDecision.CANDIDATE: "?",
    AdmissionDecision.EPISODIC:  "~",
    AdmissionDecision.REJECT:    "✗",   # heavy ballot X
}


@dataclass(frozen=True)
class AuditEntry:
    """One admission decision produced by ProtectedMemory.

    The ``__str__`` renders a compact human-readable line for
    ``memory.audit()``; the fields are structured so programs can filter,
    log, or export to JSONL without parsing text.
    """
    content: str
    source_name: str
    source_id: str | None
    trusted: bool
    functional_type: FunctionalType
    decision: AdmissionDecision
    confidence: float
    supported: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def reason(self) -> str:
        """One-line human-readable explanation of the decision."""
        d = self.decision
        t = self.functional_type.value
        who = "trusted" if self.trusted else "untrusted"
        if not self.supported:
            return "router marked item unsupported"
        return f"{t} from {who} source → {d.value}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-safe scalars only).

        The mirror-image :meth:`from_dict` reads a dict produced by this
        method back into an ``AuditEntry``; together they define the
        canonical JSONL wire format used by the ``audit_log_path=`` sink
        and the ``sourced-memory`` CLI.
        """
        return {
            "content": self.content,
            "source_name": self.source_name,
            "source_id": self.source_id,
            "trusted": self.trusted,
            "functional_type": self.functional_type.value,
            "decision": self.decision.value,
            "confidence": self.confidence,
            "supported": self.supported,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEntry":
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif ts is None:
            ts = datetime.now(timezone.utc)
        return cls(
            content=data["content"],
            source_name=data["source_name"],
            source_id=data.get("source_id"),
            trusted=bool(data["trusted"]),
            functional_type=FunctionalType(data["functional_type"]),
            decision=AdmissionDecision(data["decision"]),
            confidence=float(data.get("confidence", 1.0)),
            supported=bool(data.get("supported", True)),
            timestamp=ts,
        )

    def __str__(self) -> str:
        mark = _DECISION_MARKS.get(self.decision, "?")
        content = self.content if len(self.content) <= 60 else self.content[:57] + "..."
        return (
            f'{mark} {self.decision.value.upper():9s} "{content}"  '
            f'({self.source_name}'
            f'{":" + self.source_id if self.source_id else ""}, {self.reason})'
        )


class _ProtectedChannel:
    """Attribute-style handle returned by ``memory.<name>``. Not a public class."""

    __slots__ = ("_memory", "_name", "_trusted")

    def __init__(self, memory: "ProtectedMemory", name: str, *, trusted: bool):
        self._memory = memory
        self._name = name
        self._trusted = trusted

    def add(self, content: str, *, source_id: str | None = None,
            metadata: dict[str, Any] | None = None, **kwargs: Any) -> AuditEntry:
        return self._memory._observe(
            content=content, source=self._name, trusted=self._trusted,
            source_id=source_id, metadata=metadata, extra=kwargs,
        )

    # A memory-poisoning-defense library reads more naturally as .add than
    # .observe when used through the facade; observe stays as an alias so
    # code written against the raw Memory API still works.
    observe = add

    def session(self, source_id: str) -> "_ProtectedSessionChannel":
        return _ProtectedSessionChannel(self, source_id)


class _ProtectedSessionChannel:
    """A channel scoped to a specific source_id (for later remediation)."""

    __slots__ = ("_channel", "_source_id")

    def __init__(self, channel: _ProtectedChannel, source_id: str):
        self._channel = channel
        self._source_id = source_id

    def add(self, content: str, *, source_id: str | None = None,
            metadata: dict[str, Any] | None = None, **kwargs: Any) -> AuditEntry:
        return self._channel.add(
            content,
            source_id=source_id if source_id is not None else self._source_id,
            metadata=metadata, **kwargs,
        )

    observe = add


class ProtectedMemory:
    """The object returned by :func:`protect`. Do not instantiate directly.

    Exposes:

    - ``memory.<channel_name>.add(text, ...)`` for declared channels
    - ``.audit()`` returns the full list of :class:`AuditEntry` decisions
    - ``.beliefs()`` / ``.candidates()`` / ``.episodic()`` return the
      current admitted state
    - ``.inspect(source_id=...)`` returns everything tagged with that id
    - ``.purge(source_id=...)`` removes all state tagged with that id
    """

    def __init__(
        self,
        *,
        store: Any,
        trusted: Iterable[str],
        untrusted: Iterable[str],
        router: Router,
        policy: TrustPolicy,
        audit_log_path: str | Path | None = None,
    ):
        self._trusted_set = set(trusted)
        self._untrusted_set = set(untrusted)
        overlap = self._trusted_set & self._untrusted_set
        if overlap:
            raise ValueError(
                f"channel(s) {sorted(overlap)!r} declared as both trusted and untrusted"
            )
        self._all_declared = self._trusted_set | self._untrusted_set
        self._audit_log: list[AuditEntry] = []
        self._router = router
        self._policy = policy
        self._audit_log_path = Path(audit_log_path) if audit_log_path is not None else None
        if self._audit_log_path is not None:
            # Create the directory but not the file: an empty file is a
            # legitimate audit state and should stay valid to `sourced-memory
            # inspect` before any observations occur.
            self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Pick a backing implementation based on the store shape.
        if store is None:
            self._mode = "in_process"
            self._impl: Any = SourceAwareMemory(router=router, policy=policy)
        else:
            self._mode = "wrapped_store"
            # ``wrap_mem0`` duck-types on ``store.add(msg, user_id=, metadata=, **kw)``.
            # Any Mem0-shaped store works; Zep/custom stores can be added later
            # as their own adapter (with a compatible ``.add`` signature this
            # already works).
            self._impl = wrap_mem0(store, router=router, policy=policy)

    # ------------------------------------------------------------------
    # Attribute-style channel access
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> _ProtectedChannel:
        # Dunder + private attributes fall through to the default resolution
        # so pickle, copy, and debuggers keep working.
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._all_declared:
            return _ProtectedChannel(
                self, name, trusted=(name in self._trusted_set),
            )
        raise UnknownChannelError(
            f"channel {name!r} was not declared in protect(...). "
            f"Add it to trusted= or untrusted= at construction time. "
            f"Currently declared: trusted={sorted(self._trusted_set)}, "
            f"untrusted={sorted(self._untrusted_set)}"
        )

    def __dir__(self) -> list[str]:
        # Make declared channels visible to IDE autocomplete and dir().
        base = list(super().__dir__())
        return base + sorted(self._all_declared)

    def channel(self, name: str) -> _ProtectedChannel:
        """Explicit form of ``memory.<name>``, useful when the channel name
        is chosen at runtime."""
        return self.__getattr__(name)

    # ------------------------------------------------------------------
    # Internal: run the underlying admission and record the audit entry
    # ------------------------------------------------------------------

    def _observe(
        self,
        *,
        content: str,
        source: str,
        trusted: bool,
        source_id: str | None,
        metadata: dict[str, Any] | None,
        extra: dict[str, Any],
    ) -> AuditEntry:
        if self._mode == "in_process":
            experience = self._impl.observe(
                content, source=source, source_id=source_id,
                trusted=trusted, metadata=metadata,
            )
            # Two-phase model, hidden from the caller: consolidate immediately
            # so the caller can read back state without a second call. This
            # burns one router call per observe(), which is exactly what the
            # facade is for; users who want batched routing use the raw
            # SourceAwareMemory API.
            self._impl.consolidate()
            record = next(
                d for d in self._impl.decisions() if d.experience_id == experience.id
            )
            entry = AuditEntry(
                content=content,
                source_name=source,
                source_id=source_id,
                trusted=trusted,
                functional_type=record.functional_type,
                decision=record.decision,
                confidence=record.confidence,
                supported=record.supported,
            )
        else:
            # WrappedMem0.add returns a DecisionRecord and forwards to Mem0 iff BELIEF.
            record = self._impl.add(
                content, source=source, source_id=source_id,
                trusted=trusted, metadata=metadata, **extra,
            )
            entry = AuditEntry(
                content=content,
                source_name=source,
                source_id=source_id,
                trusted=trusted,
                functional_type=record.functional_type,
                decision=record.decision,
                confidence=record.confidence,
                supported=record.supported,
            )
        self._audit_log.append(entry)
        if self._audit_log_path is not None:
            with self._audit_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.to_dict()) + "\n")
        return entry

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    def audit(self) -> list[AuditEntry]:
        """Every admission decision this ProtectedMemory has made, in order."""
        return list(self._audit_log)

    def beliefs(self) -> list[Any]:
        """Currently admitted persistent beliefs."""
        if self._mode == "in_process":
            return self._impl.beliefs()
        # For a wrapped store, beliefs live in the store itself. Return the
        # audit entries for BELIEF decisions so the facade can still enumerate
        # what got through.
        return [e for e in self._audit_log if e.decision is AdmissionDecision.BELIEF]

    def candidates(self) -> list[Any]:
        """Untrusted evidence held aside (not admitted as a belief)."""
        if self._mode == "in_process":
            return self._impl.candidates()
        return self._impl.candidates()

    def episodic(self) -> list[Any]:
        """Transient events (not consolidated into belief)."""
        if self._mode == "in_process":
            return self._impl.episodic()
        return self._impl.episodic()

    def rejections(self) -> list[AuditEntry]:
        """All decisions that were rejected. Useful for auditing attacks."""
        return [e for e in self._audit_log if e.decision is AdmissionDecision.REJECT]

    # ------------------------------------------------------------------
    # Remediation
    # ------------------------------------------------------------------

    def inspect(self, *, source_id: str) -> dict[str, list[Any]]:
        """Return every record on this memory that is tagged with ``source_id``.

        The result is a dict with keys ``"beliefs"``, ``"candidates"``,
        ``"episodic"``, ``"audit"`` (list of :class:`AuditEntry`), and
        ``"experiences"`` (for the in-process store only).
        """
        out: dict[str, list[Any]] = {
            "beliefs": [], "candidates": [], "episodic": [],
            "experiences": [], "audit": [],
        }
        out["audit"] = [e for e in self._audit_log if e.source_id == source_id]
        if self._mode == "in_process":
            out["beliefs"] = [b for b in self._impl.beliefs() if b.source.source_id == source_id]
            out["candidates"] = [c for c in self._impl.candidates() if c.source.source_id == source_id]
            out["episodic"] = [e for e in self._impl.episodic() if e.source.source_id == source_id]
        else:
            out["candidates"] = [c for c in self._impl.candidates()
                                 if c.source.source_id == source_id]
            out["episodic"] = [e for e in self._impl.episodic()
                               if e.source.source_id == source_id]
        return out

    def purge(self, *, source_id: str) -> int:
        """Remove every record tagged with ``source_id`` from this memory.

        For an in-process ProtectedMemory this drops beliefs, candidates,
        episodic records, experiences, and decisions in one call. For a
        wrapped external store (Mem0, etc.) this drops the audit log entries
        and the candidate / episodic sidecar; whether the underlying store
        supports source-id-tagged deletion is store-specific and up to the
        caller to arrange.
        """
        removed = 0
        n = len(self._audit_log)
        self._audit_log = [e for e in self._audit_log if e.source_id != source_id]
        removed += n - len(self._audit_log)
        # Rewrite the on-disk JSONL sink so a purge is durable, not just
        # in-process. The file is created lazily by _observe, so it may not
        # exist yet on an empty ProtectedMemory.
        if self._audit_log_path is not None and self._audit_log_path.exists():
            with self._audit_log_path.open("w", encoding="utf-8") as fh:
                for entry in self._audit_log:
                    fh.write(json.dumps(entry.to_dict()) + "\n")
        if self._mode == "in_process":
            removed += self._impl.purge(source_id=source_id)
        else:
            # Purge the wrapper's own sidecars.
            n = len(self._impl._candidates)
            self._impl._candidates = [c for c in self._impl._candidates
                                      if c.source.source_id != source_id]
            removed += n - len(self._impl._candidates)
            n = len(self._impl._episodic)
            self._impl._episodic = [e for e in self._impl._episodic
                                    if e.source.source_id != source_id]
            removed += n - len(self._impl._episodic)
            n = len(self._impl._rejections)
            self._impl._rejections = [r for r in self._impl._rejections
                                      if r.source.source_id != source_id]
            removed += n - len(self._impl._rejections)
        return removed


def protect(
    store: Any = None,
    *,
    trusted: Iterable[str] | None = None,
    untrusted: Iterable[str] | None = None,
    router: Router | None = None,
    policy: TrustPolicy | None = None,
    audit_log_path: str | Path | None = None,
) -> ProtectedMemory:
    """Wrap a memory store with source-aware admission.

    Parameters
    ----------
    store:
        A Mem0-shaped memory client (any object exposing
        ``.add(msg, user_id=None, metadata=None, **kw)``), or ``None`` for an
        in-process ephemeral store. Zep, Letta, and application-owned stores
        can be used if they duck-type onto ``.add``; a dedicated adapter can
        be added later.
    trusted, untrusted:
        Lists of channel names. Every channel the caller will use must be
        declared in exactly one of these lists. Accessing an undeclared
        channel raises :class:`UnknownChannelError`.
    router:
        Content router. Defaults to :class:`~sourced_memory.router.RuleBasedRouter`
        (dependency-free). For production, pass an
        :class:`~sourced_memory.router.LLMRouter`.
    policy:
        Admission policy. Defaults to the paper's reference (source × type)
        rules.
    audit_log_path:
        Optional filesystem path. When set, every admission decision is
        appended as one JSON line, and ``memory.purge(source_id=...)``
        rewrites the file to drop matching entries. The ``sourced-memory``
        CLI reads this format for ``inspect`` / ``decisions`` / ``purge``.
    """
    return ProtectedMemory(
        store=store,
        trusted=trusted or [],
        untrusted=untrusted or [],
        router=router or RuleBasedRouter(),
        policy=policy or TrustPolicy.reference(),
        audit_log_path=audit_log_path,
    )

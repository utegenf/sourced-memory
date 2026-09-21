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
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .adapters.mem0 import WrappedMem0, wrap_mem0
from .decider import DecisionRecord
from .memory import SourceAwareMemory
from .models import AdmissionDecision, FunctionalType, Source
from .policy import TrustPolicy
from .router import Router, RuleBasedRouter


class UnknownChannelError(RuntimeError):
    """Raised when an undeclared channel is accessed on a ProtectedMemory."""


@dataclass(frozen=True)
class PurgeResult:
    """Structured result of :meth:`ProtectedMemory.purge`.

    The naming is deliberately explicit about what is a success and what
    is a gap. In particular ``records_unreachable_in_backing_store`` means
    "we could not verify these were removed from the backing store," not
    "these were successfully cleaned up locally." A security tool's output
    must not let silence or ambiguous naming stand in for verification.
    """
    records_deleted_from_backing_store: int
    records_that_failed_backing_delete: list[tuple[str, str]]  # (id, error message)
    records_unreachable_in_backing_store: int
    audit_entries_removed: int
    audit_entries_retained_for_retry: int

    @property
    def total_local_removed(self) -> int:
        """Every audit entry that this purge removed from the local log."""
        return self.audit_entries_removed

    @property
    def has_gaps(self) -> bool:
        """True iff any records were left unverified or unremoved. When True,
        the caller should NOT treat purge as complete."""
        return (
            bool(self.records_that_failed_backing_delete)
            or self.records_unreachable_in_backing_store > 0
        )

    def __str__(self) -> str:
        lines = [
            f"purged from backing store   : {self.records_deleted_from_backing_store}",
            f"audit entries removed        : {self.audit_entries_removed}",
            f"audit entries retained (retry these on next purge) : "
            f"{self.audit_entries_retained_for_retry}",
        ]
        if self.has_gaps:
            lines.append("COULD NOT VERIFY REMOVAL:")
            if self.records_unreachable_in_backing_store:
                lines.append(
                    f"  unreachable in backing store "
                    f"(no id captured on write): "
                    f"{self.records_unreachable_in_backing_store}"
                )
            if self.records_that_failed_backing_delete:
                lines.append("  delete failed, will be retried on next purge:")
                for bid, err in self.records_that_failed_backing_delete:
                    lines.append(f"    - {bid}  ({err})")
            lines.append("Some records may still exist in the backing store.")
            lines.append("Re-run this purge after fixing the underlying issue.")
            lines.append(
                "Local-only records (unreachable) will not be retried "
                "automatically; use the backing store's own tools to inspect "
                "and clean them up."
            )
        return "\n".join(lines)


class _PurgeAccumulator:
    """Mutable helper used inside ``ProtectedMemory.purge`` to build a
    :class:`PurgeResult`. Not part of the public API."""

    def __init__(self) -> None:
        self.records_deleted_from_backing_store = 0
        self.records_that_failed_backing_delete: list[tuple[str, str]] = []
        self.records_unreachable_in_backing_store = 0
        self.audit_entries_removed = 0
        self.audit_entries_retained_for_retry = 0

    def build(self) -> PurgeResult:
        return PurgeResult(
            records_deleted_from_backing_store=self.records_deleted_from_backing_store,
            records_that_failed_backing_delete=list(self.records_that_failed_backing_delete),
            records_unreachable_in_backing_store=self.records_unreachable_in_backing_store,
            audit_entries_removed=self.audit_entries_removed,
            audit_entries_retained_for_retry=self.audit_entries_retained_for_retry,
        )


class RoutingDegradedWarning(UserWarning):
    """Emitted at ``protect()`` construction when the router is weaker than
    the paper-validated LLMRouter.

    Fires on: ``protect(router=None)`` (silent default falls to
    RuleBasedRouter). Does NOT fire on ``protect(router=LLMRouter(...))``
    (paper-quality). ``protect(router=RuleBasedRouter())`` (explicit
    informed choice) emits a one-time INFO log instead, not a warning.

    The point: the security posture of the library is explicit at every
    construction site. Nobody gets a hidden default.
    """


# Application-supplied hook to authenticate a claimed source before
# admission runs. Returning False (or raising) short-circuits admission
# to REJECT before the router is called. Signature: (channel_name, content,
# source_id) -> bool.
SourceValidator = Callable[[str, str, Optional[str]], bool]


_logger = logging.getLogger("sourced_memory")
# Guarded so the explicit-RuleBasedRouter INFO fires only once per Python
# process lifetime. Long-running LangGraph agents see one line; serverless
# containers see one line per invocation, which is the intended semantics.
_explicit_rulerouter_notified = False


def _notify_router_choice(router: Router, router_was_explicit: bool) -> None:
    global _explicit_rulerouter_notified
    if not router_was_explicit:
        warnings.warn(
            "sourced-memory: running with the default keyword-only "
            "RuleBasedRouter, which misses third-person personal claims "
            "like 'the user hates X' and does not match the paper's "
            "validated behavior. To restore paper-quality routing:\n"
            "  pip install 'sourced-memory[anthropic]'\n"
            "  export ANTHROPIC_API_KEY=<key>\n"
            "  memory = protect(..., router=LLMRouter(AnthropicLLM('claude-haiku-4-5-20251001')))\n"
            "If keyword-only routing is intentional (offline demo, tests, "
            "cost-sensitive deployment), pass router=RuleBasedRouter() "
            "explicitly and this warning goes away.",
            RoutingDegradedWarning,
            stacklevel=3,
        )
        return
    if isinstance(router, RuleBasedRouter) and not _explicit_rulerouter_notified:
        _explicit_rulerouter_notified = True
        _logger.info(
            "sourced-memory: using RuleBasedRouter explicitly (keyword-only "
            "routing). This misses third-person personal claims like "
            "'the user hates X'. For paper-quality routing use LLMRouter."
        )


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
    # Populated when a decision short-circuits the router (e.g. via a
    # rejecting source_validator hook). ``reason`` returns this verbatim
    # when it is set, so operators see the actual failure cause.
    short_circuit_reason: str | None = None
    # For BELIEF decisions written to a wrapped external store, the id the
    # underlying store returned. None for in-process ProtectedMemory (which
    # owns storage) and for BELIEFs where the underlying store's ``.add()``
    # returned no recognizable id. Used by ``ProtectedMemory.purge`` to
    # loop-delete the backing-store record.
    backing_store_id: str | None = None

    @property
    def reason(self) -> str:
        """One-line human-readable explanation of the decision."""
        if self.short_circuit_reason is not None:
            return self.short_circuit_reason
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
            "short_circuit_reason": self.short_circuit_reason,
            "backing_store_id": self.backing_store_id,
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
            short_circuit_reason=data.get("short_circuit_reason"),
            backing_store_id=data.get("backing_store_id"),
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
        source_validator: SourceValidator | None = None,
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
        self._source_validator = source_validator
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
        # source_validator: application-supplied hook. Runs BEFORE the router,
        # so a rejection here saves the LLM call for cost/latency-sensitive
        # deployments. A False return or a raised exception both short-circuit
        # to REJECT; the reason surfaces in the audit log for operator review.
        if self._source_validator is not None:
            try:
                allowed = self._source_validator(source, content, source_id)
            except Exception as exc:
                return self._record_short_circuit_reject(
                    content=content, source=source, source_id=source_id,
                    trusted=trusted,
                    reason=f"source_validator raised {type(exc).__name__}: {exc}",
                )
            if not allowed:
                return self._record_short_circuit_reject(
                    content=content, source=source, source_id=source_id,
                    trusted=trusted,
                    reason="source_validator rejected",
                )
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
                backing_store_id=record.backing_store_id,
            )
        self._audit_log.append(entry)
        if self._audit_log_path is not None:
            with self._audit_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.to_dict()) + "\n")
        return entry

    def _record_short_circuit_reject(
        self, *, content: str, source: str, source_id: str | None,
        trusted: bool, reason: str,
    ) -> AuditEntry:
        """Emit a REJECT AuditEntry without invoking the router or the
        underlying admission impl. Used by the source_validator short-circuit
        so a rejecting validator saves the router call and the router's cost.
        The reason string is preserved on the entry via metadata for audit UX.
        """
        entry = AuditEntry(
            content=content,
            source_name=source,
            source_id=source_id,
            trusted=trusted,
            functional_type=FunctionalType.EXTERNAL_FACT,  # nominal; unused for REJECT
            decision=AdmissionDecision.REJECT,
            confidence=1.0,
            supported=False,
            short_circuit_reason=reason,
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

    def purge(self, *, source_id: str) -> PurgeResult:
        """Remove every record tagged with ``source_id`` from this memory.

        In-process ProtectedMemory: drops beliefs, candidates, episodic
        records, experiences, and decisions in one call. Complete removal.

        Wrapped external store (Mem0, etc.): iterates every audit entry
        whose ``decision`` is BELIEF and whose ``backing_store_id`` is set,
        calling ``store.delete(backing_store_id)`` on each. Successes are
        removed from the audit log. **Failures are retained in the audit
        log so a subsequent purge can retry them.** This means a permanent
        orphan (backing-store record with no retry hook) is impossible as
        long as the id was captured on write. Local-only entries (BELIEF
        with backing_store_id=None) count under
        ``records_unreachable_in_backing_store``: we cannot verify their
        removal from the backing store.

        Not concurrency-safe: running ``purge(source_id=X)`` concurrently
        with an in-flight write for the same source_id can leave an orphan
        in the backing store that is invisible to future purges. Callers
        must serialize purge against writes for a given source_id. A
        future ``purge_and_freeze`` will close this race by revoking the
        source_id first, but is out of scope for 0.1.0.
        """
        result = _PurgeAccumulator()

        # In-process branch: sidecars are ours, everything is deterministic.
        if self._mode == "in_process":
            impl_removed = self._impl.purge(source_id=source_id)
            n = len(self._audit_log)
            self._audit_log = [e for e in self._audit_log if e.source_id != source_id]
            result.audit_entries_removed = n - len(self._audit_log)
            self._rewrite_audit_log_if_configured()
            # Attribute impl_removed to the local counter; there is no
            # backing store on this path.
            result.audit_entries_removed = max(result.audit_entries_removed, impl_removed)
            return result.build()

        # Wrapped-store branch: retry-safe loop-delete against the backing store.
        retained: list[AuditEntry] = []
        for entry in self._audit_log:
            if entry.source_id != source_id:
                retained.append(entry)
                continue
            if entry.decision is not AdmissionDecision.BELIEF:
                # Non-BELIEF decisions were never written to the backing store.
                # Just drop the audit entry.
                result.audit_entries_removed += 1
                continue
            if entry.backing_store_id is None:
                # BELIEF, but we do not have an id to delete against. Drop
                # the audit entry but count the gap under "unreachable".
                result.records_unreachable_in_backing_store += 1
                result.audit_entries_removed += 1
                continue
            # BELIEF with a captured id: try to delete from the backing store.
            try:
                self._impl.mem0.delete(entry.backing_store_id)
            except Exception as e:
                # Retain the audit entry so a later purge can retry.
                retained.append(entry)
                result.audit_entries_retained_for_retry += 1
                result.records_that_failed_backing_delete.append(
                    (entry.backing_store_id, f"{type(e).__name__}: {e}")
                )
                continue
            result.records_deleted_from_backing_store += 1
            result.audit_entries_removed += 1

        self._audit_log = retained
        self._rewrite_audit_log_if_configured()

        # Sidecars on the wrapper: candidates / episodic / rejections. Purge
        # unconditionally; they are ours, not the backing store's.
        for attr in ("_candidates", "_episodic", "_rejections"):
            bucket = getattr(self._impl, attr)
            filtered = [r for r in bucket if r.source.source_id != source_id]
            setattr(self._impl, attr, filtered)
            # These do not add to the "unreachable" count; they were never
            # written to the backing store to begin with.
        return result.build()

    def _rewrite_audit_log_if_configured(self) -> None:
        """Rewrite the on-disk JSONL sink to match the in-memory audit log."""
        if self._audit_log_path is not None and self._audit_log_path.exists():
            with self._audit_log_path.open("w", encoding="utf-8") as fh:
                for entry in self._audit_log:
                    fh.write(json.dumps(entry.to_dict()) + "\n")


def protect(
    store: Any = None,
    *,
    trusted: Iterable[str] | None = None,
    untrusted: Iterable[str] | None = None,
    router: Router | None = None,
    policy: TrustPolicy | None = None,
    audit_log_path: str | Path | None = None,
    source_validator: SourceValidator | None = None,
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
        Content router. If omitted, defaults to
        :class:`~sourced_memory.router.RuleBasedRouter` and emits a
        :class:`RoutingDegradedWarning` at construction; the keyword-only
        router misses third-person personal claims and does not match the
        paper's validated behavior. For paper-quality routing pass an
        :class:`~sourced_memory.router.LLMRouter` explicitly. To silence
        the warning without upgrading, pass ``router=RuleBasedRouter()``
        explicitly to signal an informed choice; a one-time INFO log names
        the tradeoff.
    policy:
        Admission policy. Defaults to the paper's reference (source × type)
        rules.
    audit_log_path:
        Optional filesystem path. When set, every admission decision is
        appended as one JSON line, and ``memory.purge(source_id=...)``
        rewrites the file to drop matching entries. The ``sourced-memory``
        CLI reads this format for ``inspect`` / ``decisions`` / ``purge``.
    source_validator:
        Optional application-supplied hook, ``(channel_name, content,
        source_id) -> bool``. Called before the router runs; returning
        ``False`` or raising an exception short-circuits admission to
        REJECT (saving the router call and its cost). The library assumes
        source metadata is honestly supplied by default; this hook is
        where an application plugs in its own source authentication
        (signature check, network origin, session token) to close the
        source-spoofing gap. Failing loudly is the right default here.
    """
    router_was_explicit = router is not None
    if router is None:
        router = RuleBasedRouter()
    _notify_router_choice(router, router_was_explicit)
    return ProtectedMemory(
        store=store,
        trusted=trusted or [],
        untrusted=untrusted or [],
        router=router,
        policy=policy or TrustPolicy.reference(),
        audit_log_path=audit_log_path,
        source_validator=source_validator,
    )

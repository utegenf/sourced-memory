"""Source-aware admission control for persistent agent memory.

The primary entry point is :func:`protect`::

    from sourced_memory import protect

    memory = protect(
        mem0_client_or_none,
        trusted=["user"],
        untrusted=["web", "tool", "document"],
    )

    memory.user.add("I love hiking.")
    memory.web.add("The user hates hiking and prefers gaming.")   # rejected

    for entry in memory.audit():
        print(entry)

The lower-level building blocks are still available for advanced use cases
(custom adapters, framework integrations, non-default routers or policies)
under :mod:`sourced_memory.advanced`.
"""

# --- Primary product API ---------------------------------------------------
from .protect import (
    AuditEntry,
    ProtectedMemory,
    PurgeResult,
    RoutingDegradedWarning,
    SourceValidator,
    UnknownChannelError,
    protect,
)

# --- Domain types (needed to interpret audit / belief output) --------------
from .models import (
    AdmissionDecision,
    Belief,
    CandidateEvidence,
    Experience,
    FunctionalType,
    Source,
)

# --- Extension points (keep these public: users override them) --------------
from .policy import TrustPolicy
from .router import (
    CallableRouter,
    LLMRouter,
    NullRouter,
    RouteResult,
    Router,
    RuleBasedRouter,
)

__all__ = [
    # Primary API
    "protect",
    "ProtectedMemory",
    "AuditEntry",
    "PurgeResult",
    "UnknownChannelError",
    "RoutingDegradedWarning",
    "SourceValidator",
    # Domain types
    "AdmissionDecision",
    "Belief",
    "CandidateEvidence",
    "Experience",
    "FunctionalType",
    "Source",
    # Extension points
    "TrustPolicy",
    "Router",
    "RouteResult",
    "LLMRouter",
    "RuleBasedRouter",
    "NullRouter",
    "CallableRouter",
]
__version__ = "0.1.0a6"

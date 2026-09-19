"""Source-aware admission control for persistent agent memory."""

from .decider import Decider, DecisionRecord
from .memory import SourceAwareMemory
from .models import (
    AdmissionDecision,
    Belief,
    CandidateEvidence,
    Experience,
    FunctionalType,
    Source,
)
from .policy import SourceTypePolicy
from .router import (
    CallableRouter,
    LLMRouter,
    NullRouter,
    RouteResult,
    Router,
    RuleBasedRouter,
)

# Stable short aliases: sourced_memory.Memory / TrustPolicy are the names used
# in the README quickstart and adapters. Not deprecated.
Memory = SourceAwareMemory
TrustPolicy = SourceTypePolicy

__all__ = [
    "AdmissionDecision",
    "Belief",
    "CallableRouter",
    "CandidateEvidence",
    "Decider",
    "DecisionRecord",
    "Experience",
    "FunctionalType",
    "LLMRouter",
    "Memory",
    "NullRouter",
    "RouteResult",
    "Router",
    "RuleBasedRouter",
    "Source",
    "SourceAwareMemory",
    "SourceTypePolicy",
    "TrustPolicy",
]
__version__ = "0.1.0a1"

"""Source-aware admission control for persistent agent memory."""

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

# Temporary compatibility aliases while the v0 API is being stabilized.
Memory = SourceAwareMemory
TrustPolicy = SourceTypePolicy

__all__ = [
    "AdmissionDecision",
    "Belief",
    "CandidateEvidence",
    "Experience",
    "FunctionalType",
    "Memory",
    "Source",
    "SourceAwareMemory",
    "SourceTypePolicy",
    "TrustPolicy",
]
__version__ = "0.1.0"

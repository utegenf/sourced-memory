"""Core data models for source-aware belief updating."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional
from uuid import uuid4

class FunctionalType(str, Enum):
    PERSONAL_PREFERENCE = "personal_preference"
    GENERAL_RULE = "general_rule"
    RELATIONAL_FACT = "relational_fact"
    EXTERNAL_FACT = "external_fact"
    EVENT = "event"

class AdmissionDecision(str, Enum):
    BELIEF = "belief"
    CANDIDATE = "candidate"
    EPISODIC = "episodic"
    REJECT = "reject"

@dataclass(frozen=True)
class Source:
    """Application-supplied provenance. It is never inferred from claim text."""
    name: str
    source_id: Optional[str] = None
    trusted: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass
class Experience:
    content: str
    source: Source
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    episode_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class Belief:
    content: str
    functional_type: FunctionalType
    source: Source
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    experience_id: Optional[str] = None
    confidence: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class CandidateEvidence:
    content: str
    functional_type: FunctionalType
    source: Source
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    experience_id: Optional[str] = None
    confidence: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

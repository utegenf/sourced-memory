# Architecture

This document defines the v0 public architecture for the reusable source-aware memory layer. The research implementation remains separate: it may use richer LLM prompts, experiment-specific stores, and paper-specific consolidation logic without expanding the core API.

## Design boundary

The library separates three concerns:
1. **Content interpretation** — determine what functional type an experience has.
2. **Source-conditioned admission** — decide whether that experience becomes a belief, candidate evidence, episodic memory, or is rejected.
3. **State management** — store and return the resulting objects.

Provenance is **application-supplied metadata**. The library never infers trust or source identity from message content.

## Public protocols

The following are the v0 protocol contracts. Implementations are intentionally omitted here; the concrete library classes may provide the default implementations.

```python
from typing import Any, Iterable, Mapping, Protocol, Sequence

from sourced_memory.models import AdmissionDecision, Belief, CandidateEvidence, Experience, FunctionalType, Source
from sourced_memory.router import RouteResult

class Router(Protocol):
    """Interpret content without access to provenance. Structurally, this
    signature takes a ``str`` — a router implementation cannot see the item's
    source even by mistake."""
    def route(self, content: str) -> RouteResult: ...

class Policy(Protocol):
    """Map source + functional type to an admission decision."""
    def decide(self, *, source: Source, functional_type: FunctionalType, experience: Experience) -> AdmissionDecision: ...

class Backend(Protocol):
    """Store memory state; implementations may be in-memory or external."""
    def add_belief(self, belief: Belief) -> None: ...
    def add_candidate(self, candidate: CandidateEvidence) -> None: ...
    def add_experience(self, experience: Experience) -> None: ...
    def beliefs(self) -> Sequence[Belief]: ...
    def candidates(self) -> Sequence[CandidateEvidence]: ...
    def episodic(self) -> Sequence[Experience]: ...

class LLM(Protocol):
    """Minimal provider-neutral interface for LLM-backed routers/consolidators."""
    def complete(self, *, system: str, user: str, response_schema: type[Any] | None = None) -> str | Mapping[str, Any]: ...

def consolidate(buffer: Iterable[Experience], *, policy: Policy, router: Router) -> Sequence[Belief | CandidateEvidence | Experience]:
    """Pure consolidation contract: route buffered experiences and apply policy."""
    ...
```

## v0 components

### `Memory`

The high-level integration object owns a buffer and a backend. `observe()` records an experience and returns the buffered `Experience`; it does **not** call the router, mutate persistent beliefs, or consolidate automatically. `consolidate()` is an explicit operation, and reader methods (`beliefs()`, `candidates()`, `episodic()`, `decisions()`) return the current state without triggering consolidation. `pending()` reports how many buffered experiences have not yet been consolidated.

### Router implementations

v0 exposes three router modes:
- **LLMRouter** — pluggable LLM-backed content classifier, including the paper's implementation.
- **RuleRouter** — dependency-free keyword/regex baseline for deterministic local use.
- **NullRouter** — for applications that already provide a functional type.

The router receives content, not source metadata.

### Policy

Policies are configuration rather than hardcoded application logic. The reference policy expresses source × functional-type rules, but applications may supply their own `Policy` implementation.

### Backend

v0 ships only an in-memory backend. The `Backend` protocol is the extension point for Redis, Postgres, vector databases, application-owned stores, or other persistence systems. SQLite is deliberately out of scope for v0.

### LLM

The core package has no mandatory model-provider dependency. Anthropic and OpenAI integrations belong behind optional extras and adapters.

## State model

The library exposes four explicit destinations:
- **Belief** — admitted persistent belief.
- **Candidate** — evidence retained for possible future verification/promotion.
- **Episodic** — event-like experience retained without changing persistent beliefs.
- **Rejected** — an admission decision retained in the operation result/log rather than promoted to belief state.

## Explicit v0 non-goals

v0 does **not** ship:
- a SQLite or other persistent storage implementation;
- automatic corroboration or candidate-to-belief promotion;
- learned multi-tier trust or source reputation;
- a hosted memory service;
- non-Python language bindings;
- automatic consolidation on `observe()`;
- a mandatory LLM dependency;
- framework-specific integrations as a requirement for the core package.

## Research separation

The paper's experiments are reproducibility artifacts, not the public API contract. The research layer may preserve experiment-specific classes and metrics while importing shared policy/model definitions where doing so does not alter the experimental protocol.
# Architecture

`sourced-memory` sits **in front of** an application's existing memory
system and decides which incoming experiences may modify persistent
belief. It does not own durable storage; storage stays with whatever the
application uses (Mem0, Zep, custom DB, an in-process list).

## Design boundary

The library separates three concerns:

1. **Content interpretation.** A router classifies each experience into a
   functional type (personal preference, general rule, relational fact,
   external fact, event). The router sees the item text and nothing else.
2. **Source-conditioned admission.** A policy maps `(source, functional
   type) → destination`. This is where the trust decision happens; the
   router does not participate in it.
3. **State management.** The admitted item is forwarded to the backing
   memory system (for `BELIEF`), retained in a per-instance sidecar (for
   `CANDIDATE` and `EPISODIC`), or dropped (for `REJECT`). All decisions
   are recorded in an audit log.

Provenance is **application-supplied metadata**. The library never infers
trust or source identity from message content. That is exactly the
laundering vector the mechanism exists to prevent.

## The path a single experience takes

```
        Application  (user message, tool output, retrieved document, ...)
              │
              ▼
       ┌──────────────┐
       │  Channel     │ ← declared once at protect() setup with a name and a trust flag
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │   Router     │ ← sees ONLY the content string
       └──────┬───────┘
              │
              ▼ functional type
       ┌──────────────┐
       │   Policy     │ ← sees ONLY (source, functional type)
       └──────┬───────┘
              │
              ▼
    ┌─────────┴─────────┬──────────────┬────────────┐
    ▼                   ▼              ▼            ▼
 BELIEF             CANDIDATE       EPISODIC      REJECT
    │                   │              │            │
    ▼                   ▼              ▼            ▼
Existing memory  Held in-process  Held in-process  Audit log only
(Mem0, Zep, ...)   sidecar          sidecar
```

## Extension points

Three protocols make up the entire extension surface:

```python
from typing import Protocol
from sourced_memory.models import AdmissionDecision, FunctionalType, Source
from sourced_memory.router import RouteResult

class Router(Protocol):
    """Interpret content without access to provenance. The signature takes
    a ``str`` so a router implementation cannot see the item's source even
    by mistake."""
    def route(self, content: str) -> RouteResult: ...


class Policy(Protocol):
    """Map (source, functional type) to an admission decision."""
    def decide(self, source: Source, functional_type: FunctionalType) -> AdmissionDecision: ...


class LLM(Protocol):
    """Minimal provider-neutral interface for LLM-backed routers."""
    def complete_json(self, *, system: str, user: str,
                      response_schema: dict, response_name: str) -> dict: ...
```

Concrete implementations shipped in `0.1.x`:

- **Routers.** `LLMRouter` (production, needs an `LLM`), `RuleBasedRouter`
  (keyword-only, no LLM), `NullRouter` (passthrough, use when the caller
  already knows the type), `CallableRouter` (any callable that returns a
  `RouteResult`).
- **Policy.** `TrustPolicy` is the `(source × functional-type)` rule
  table. Use `TrustPolicy.reference()` for the paper's default rules or
  construct with your own mapping.
- **LLM providers.** `MockLLM` (in core, no keys), `AnthropicLLM` (direct
  API or Bedrock, via `[anthropic]` extra), `OpenAILLM` (via `[openai]`
  extra). More can be added by anyone implementing the `LLM` protocol.

## Adapters

Adapters bridge `sourced-memory` to a specific memory system. `wrap_mem0`
is the shipped reference integration; any object that exposes a
Mem0-shaped `.add(msg, user_id=None, metadata=None, **kw)` works today via
`protect(store, ...)`. Framework-specific adapters (LangGraph, Zep,
Letta) live in `examples/` first; a package under
`sourced_memory.frameworks.*` is created only when a real user asks.

## What `protect()` actually does

`protect()` composes the pieces above and hides the two-phase
observe/consolidate split of the raw `SourceAwareMemory`. For an
in-process backing store it batches to a single consolidation on each
`observe()`; for a wrapped external store it decides admission inline
and forwards to `.add()` iff the decision is `BELIEF`. Both paths append
to the same audit log, so `memory.audit()` and `memory.purge(source_id=)`
behave the same regardless of backend.

## Deliberately not in v0

- No persistent storage; bring your own store or accept the ephemeral
  in-process default. A formal `Backend` protocol may appear once the
  audit log needs one; for now the audit log is in-process too.
- No corroboration or candidate-to-belief promotion.
- No hosted service.
- No framework-specific integration packages; examples first, packages
  when a real user needs them.
- No mandatory LLM dependency; `RuleBasedRouter` covers the offline demo
  and `MockLLM` covers tests.

## Advanced primitives

For custom adapters, framework integrations, runtime-dynamic channel
sets, or reproducing the paper's exact two-phase model, import the raw
building blocks from `sourced_memory.advanced`:

```python
from sourced_memory.advanced import (
    SourceAwareMemory,     # in-process, two-phase (observe / consolidate)
    Channel,               # standalone channel object
    Decider,               # storage-free decision function
    wrap_mem0,             # Mem0 adapter directly
)
```

## Research separation

The paper's experiments live under `research/` and are reproducibility
artifacts, not the public API contract. The research code carries a
`SleepAgent` and richer prompts tuned to the paper's benchmark; the
library sits at a higher altitude. `docs/research.md` links the two.

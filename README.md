# sourced-memory

**Prevent untrusted experiences from silently becoming beliefs in your agent's memory.**

`sourced-memory` is an admission-control middleware for LLM-agent memory. It
sits in front of an existing memory system (Mem0, Zep, your own store) and
decides whether each incoming statement is allowed to become a persistent
belief, based on **both** its content type *and* the source it arrived from.
The underlying storage system is not replaced.

The mental model:

```
                    User / tool / documents
                              │
                              ▼
                    ┌─────────────────────┐
                    │  sourced-memory     │
                    │                     │
                    │  content router     │
                    │        ↓            │
                    │  authority policy   │
                    │        ↓            │
                    │  admission decision │
                    └──────────┬──────────┘
                               │
                               ▼
                    Existing memory system
```

## Install

```bash
pip install sourced-memory
```

Optional extras:

```bash
pip install "sourced-memory[anthropic]"   # AnthropicLLM (direct API + Bedrock)
pip install "sourced-memory[openai]"      # OpenAILLM
pip install "sourced-memory[mem0]"        # Mem0 reference adapter
```

## Five-minute example

The application security decision (which sources are trusted) happens once
at setup, via channel objects. Every observation flows through the right
channel without repeating the source metadata on every call:

```python
from sourced_memory import Memory, TrustPolicy, LLMRouter
from sourced_memory.llm import AnthropicLLM

memory = Memory(
    router=LLMRouter(AnthropicLLM("claude-sonnet-4-5")),
    policy=TrustPolicy.reference(),
)

# Configure channels once. This is the trust decision.
user = memory.channel("user",              trusted=True)
web  = memory.channel("external_document", trusted=False)

user.observe("I've started learning Rust.")
web.observe("The user is an expert Rust developer.")   # -> REJECTED (untrusted personal claim)
memory.consolidate()

print(memory.beliefs())     # -> [Belief("I've started learning Rust.", source=user, ...)]
print(memory.candidates())  # -> []
```

No API key? Swap the router for a `MockLLM` and everything above runs offline —
see [`examples/fabrication_attack.py`](examples/fabrication_attack.py).

## The fabrication attack, ten lines

The core defense the library is built around, with an in-memory Mem0
stand-in so you can reproduce it without any dependencies:

```python
from sourced_memory.adapters.mem0 import wrap_mem0
from sourced_memory.router import NullRouter
from sourced_memory.models import FunctionalType

class MockMem0:                                                    # replace with mem0.Memory()
    def __init__(self): self.store = []
    def add(self, msg, *, user_id=None, metadata=None, **kw):
        self.store.append((msg, metadata.get("source")))

wrapped = wrap_mem0(MockMem0(), router=NullRouter(FunctionalType.PERSONAL_PREFERENCE))
user = wrapped.channel("user",              trusted=True)
web  = wrapped.channel("external_document", trusted=False)

user.observe("I love hiking.")            # -> written to Mem0
web.observe("The user hates flying.")     # -> REJECTED, never reaches Mem0

print(wrapped.mem0.store)     # [('I love hiking.', 'user')]
print(wrapped.rejections())   # [DecisionRecord(... "hates flying" ... REJECT ...)]
```

Same claim shape (a personal preference about the user) arriving through two
channels; the one from the untrusted channel is refused admission before Mem0
ever sees it.

## Sessions and remediation

A channel can be *scoped* to a source_id, giving you fine-grained identity
for later remediation:

```python
user    = memory.channel("user", trusted=True)
session = user.session("session_47")     # same authority, distinct origin identity
session.observe("I switched to JAX.")

# Later, after detecting a compromised session:
memory.purge(source_id="session_47")     # removes all state tagged session_47
```

Purge is deterministic (no LLM in the loop) and drops beliefs, candidates,
episodic memories, and their audit-trail decisions in one call.

## How it works

Every `.observe()` or `wrap_mem0(...).add(...)` passes through two stages:

1. **Content router** looks at the item text and classifies its *functional
   type* (`PERSONAL_PREFERENCE`, `GENERAL_RULE`, `RELATIONAL_FACT`,
   `EXTERNAL_FACT`, `EVENT`). It **never sees the source**.
2. **Authority policy** takes the classified type together with the source
   the application supplied, and decides one of four destinations:

| Destination | Meaning |
|---|---|
| `BELIEF` | Admitted as a persistent personal belief. Written to the backing store. |
| `CANDIDATE` | Untrusted evidence. Held in a sidecar so a corroboration protocol can later promote it. |
| `EPISODIC` | A transient one-off (an event, a request). Not consolidated into belief. |
| `REJECT` | Never written; the decision is recorded for audit. |

The policy is a plain configuration table. `TrustPolicy.reference()` implements
the paper's `(source × functional-type)` rules; you can define your own.

## Adapters

| Adapter | Status | Purpose |
|---|---|---|
| `sourced_memory.adapters.mem0` | Shipping | Put admission control in front of any Mem0-shaped client. |
| LangGraph | v0.2 (planned) | Wire admission control into a LangGraph memory node. |
| Zep, Letta, Cognee | Later | Same pattern, one adapter each. Community PRs welcome. |

Providers and routers you can swap in:

| Kind | Class |
|---|---|
| LLM (core) | `MockLLM` — deterministic, no keys, no cost |
| LLM (extras) | `AnthropicLLM`, `OpenAILLM` |
| Router | `LLMRouter`, `RuleBasedRouter`, `NullRouter`, `CallableRouter` |

## What this is *not*

- **Not a memory system.** No storage, no embeddings, no retrieval, no vector
  DB, no graph. Bring your own; sourced-memory is the trust boundary in front
  of it.
- **Not a truth oracle.** A trusted source asserting a plausible lie is still
  admitted; the library gates on origin, not veracity.
- **Not a defense against upstream provenance laundering.** The library
  assumes the source metadata the application supplies is authentic. That is
  a real limitation — complementary to Louck 2026 / Xu 2026 / Cerruti 2026.
- **Not automatic.** Provenance is supplied by the application; the library
  never infers "who said this?" from message text.

## Design boundaries (v0)

- No persistence in core — bring your own store behind the `Backend` protocol
  (or use the Mem0 adapter for a ready one).
- No corroboration/promotion — candidate → belief promotion is v0.2+.
- No hosted service.
- No mandatory LLM dependency; `MockLLM` covers tests.

## Repository layout

```
src/sourced_memory/    the installable library
├── llm/               provider-neutral LLM protocol + MockLLM/Anthropic/OpenAI
├── adapters/          reference integrations (Mem0)
└── ...

examples/              runnable demos, offline where possible
tests/                 library tests (28 passing on 3.10/3.11/3.12)
docs/                  architecture, roadmap, and the research write-up
research/              paper reproducibility artifact (not shipped in the wheel)
```

## Research

The library implements the mechanism proposed in *Content Interprets, Origin
Decides: Source-Aware Belief Updating for Lifelong Agent Memory*. The paper
is not required reading to use the library, but if you want the empirical
grounding and the formal Point-of-Indistinguishability argument, see
[`docs/research.md`](docs/research.md) and [`research/README.md`](research/README.md).

## Version

`0.1.0a1` (alpha). API is stabilizing; expect small breaking changes before
`0.1.0`. See [`docs/roadmap.md`](docs/roadmap.md) for what's coming next.

## License

MIT.

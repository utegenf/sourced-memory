# sourced-memory

**Stop untrusted inputs from silently becoming beliefs in your agent's memory.**

Modern LLM agents keep persistent memory (Mem0, Zep, Letta, your own store).
That memory is written to by anything the agent sees: user messages, tool
outputs, retrieved documents, sub-agent replies. Content-only memory
systems cannot tell whether a plausible personal statement came from the
user or from a poisoned web page. `sourced-memory` sits in front of any
memory store and decides which experiences are allowed to become persistent
beliefs, based on both what the content is and where it came from.

```
              User          Tool output          Retrieved documents
                │                │                        │
                └────────────────┼────────────────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │     sourced-memory     │
                     │                        │
                     │  content router        │
                     │        ↓               │
                     │  authority policy      │
                     │        ↓               │
                     │  admission decision    │
                     └───────────┬────────────┘
                                 │
                       ┌─────────┼─────────┐
                       ▼         ▼         ▼
                    BELIEF   CANDIDATE   REJECT
                       │
                       ▼
              Existing memory system
              (Mem0, Zep, custom DB, ...)
```

## Install

```bash
pip install sourced-memory
```

Optional extras:

```bash
pip install "sourced-memory[anthropic]"   # AnthropicLLM (direct API + Bedrock)
pip install "sourced-memory[openai]"      # OpenAILLM
```

## 30-second demo

Fully offline. No API keys, no external memory, no LLM. Copy-paste into a
Python REPL after `pip install sourced-memory`:

```python
from sourced_memory import protect

memory = protect(
    trusted   = ["user"],
    untrusted = ["web", "tool", "document"],
)

memory.user.add("I love hiking.")
memory.web.add("The user hates hiking and prefers gaming.")

for entry in memory.audit():
    print(entry)
```

Output:

```
✓ BELIEF    "I love hiking."                                (user, personal_preference from trusted source → belief)
? CANDIDATE "The user hates hiking and prefers gaming."     (web, external_fact from untrusted source → candidate)
```

The web page's claim about the user never becomes a belief. Same claim
shape, two origins; only the trusted-source one is admitted.

## Two lines to add source-aware admission to Mem0

```python
from mem0 import Memory as Mem0Memory
from sourced_memory import protect

memory = protect(
    Mem0Memory(),
    trusted   = ["user"],
    untrusted = ["web", "tool", "document"],
)

memory.user.add("I've started learning Rust.",              user_id="alice")
memory.web.add("The user is an expert Rust developer.",      user_id="alice")    # rejected
memory.document.add("Paris is the capital of France.",       user_id="alice")   # held as candidate
```

Only the first line reaches Mem0. The second is refused admission before
Mem0 sees it; the third is held as candidate evidence (untrusted world
fact, not a personal belief). Any Mem0-shaped store (Zep, Letta, your own
`.add(...)` object) works the same way.

## Sessions and remediation

Every observation can be scoped to an identity you can revoke later:

```python
session = memory.user.session("session_47")   # same authority, scoped source_id
session.add("I switched to JAX.")

# ... later, if session 47 turns out to be compromised:
found = memory.inspect(source_id="session_47")   # everything tagged with it
removed = memory.purge(source_id="session_47")   # deterministic drop
```

`purge()` is a pure filter; no LLM in the loop. It drops beliefs,
candidates, episodic records, and their audit entries in one call.

## What each destination means

Every admission produces exactly one of four outcomes:

| Destination     | Meaning                                                                                        |
|-----------------|------------------------------------------------------------------------------------------------|
| `BELIEF`        | Admitted as a persistent personal belief; written to the backing store.                        |
| `CANDIDATE`     | Untrusted evidence; retained in a sidecar but never surfaced as a belief.                      |
| `EPISODIC`      | Transient one-off (an event, a request); not consolidated into belief.                         |
| `REJECT`        | Never written; the decision is recorded in the audit log for review.                           |

## Undeclared channels are an error, not a default

Access to a channel that was not declared in `protect(...)` raises
`UnknownChannelError`:

```python
memory = protect(trusted=["user"], untrusted=["web"])
memory.slack.add("...")     # -> UnknownChannelError
```

Loud beats silent for a security library; the caller must declare every
source it will use. Downgrading an unknown source to "untrusted" would be
convenient and dangerous.

## Swap the router, swap the policy

`protect()` defaults to a dependency-free `RuleBasedRouter` (keyword
classification) and the paper's reference `(source × functional-type)`
policy. Both are overridable:

```python
from sourced_memory import protect, TrustPolicy
from sourced_memory.router import LLMRouter
from sourced_memory.llm import AnthropicLLM

memory = protect(
    mem0_client,
    trusted   = ["user"],
    untrusted = ["web", "tool"],
    router    = LLMRouter(AnthropicLLM("claude-sonnet-4-5")),
    policy    = TrustPolicy.reference(),
)
```

## Advanced: raw primitives

Everything above is a thin facade over composable pieces. When you need
finer control (custom adapters, framework integrations, runtime-dynamic
channels, or the paper's exact two-phase model), the primitives live in
`sourced_memory.advanced`:

```python
from sourced_memory.advanced import (
    SourceAwareMemory,     # in-process, two-phase (observe / consolidate)
    Channel,               # standalone channel object
    Decider,               # storage-free decision function
    wrap_mem0,             # Mem0 adapter directly
)
```

The primary API (`protect`, `AuditEntry`, `UnknownChannelError`, plus
`Router`, `LLMRouter`, `TrustPolicy`, etc.) stays at the top level.

## What this is not

- **Not a memory system.** No storage, no embeddings, no retrieval, no
  vector DB. Bring your own; sourced-memory is the trust boundary in
  front of it.
- **Not a truth oracle.** A trusted source asserting a plausible lie is
  still admitted; the library gates on origin, not veracity.
- **Not a defense against upstream provenance laundering.** The library
  assumes the source metadata the application supplies is authentic.
  Complementary to work like Louck 2026 and Xu 2026 on non-malleable
  origin binding.
- **Not automatic.** Provenance is supplied by the application; the
  library never infers "who said this?" from message text.

## Design boundaries (v0.1)

- No persistence in core; bring your own store.
- No corroboration/promotion (candidate → belief promotion is planned for
  a later release).
- No hosted service.
- No mandatory LLM dependency.

## Version

`0.1.0a4` (alpha). API stabilizing; small breaking changes remain possible
before `0.1.0`. See [`docs/roadmap.md`](docs/roadmap.md) for what's next.

## Research

The library implements the mechanism proposed in *Content Interprets,
Origin Decides: Source-Aware Belief Updating for Lifelong Agent Memory*.
The paper is not required reading to use the library; see
[`docs/research.md`](docs/research.md) if you want the empirical grounding
and the formal Point-of-Indistinguishability argument.

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

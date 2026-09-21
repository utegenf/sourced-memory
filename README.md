# sourced-memory

[![CI](https://github.com/utegenf/sourced-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/utegenf/sourced-memory/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/sourced-memory.svg)](https://pypi.org/project/sourced-memory/)
[![Python](https://img.shields.io/pypi/pyversions/sourced-memory.svg)](https://pypi.org/project/sourced-memory/)
[![License](https://img.shields.io/pypi/l/sourced-memory.svg)](LICENSE)

**Stop untrusted inputs from silently becoming beliefs in your agent's memory.**

Modern LLM agents keep persistent memory (Mem0, Zep, Letta, your own store).
That memory is written to by anything the agent sees: user messages, tool
outputs, retrieved documents, sub-agent replies. Content-only memory
systems cannot tell whether a plausible personal statement came from the
user or from a poisoned web page. `sourced-memory` sits in front of any
memory store and decides which experiences are allowed to become persistent
beliefs, based on both what the content is and where it came from.

> **sourced-memory does not own durable storage.** It protects writes into
> the application's existing memory system. Storage stays with whatever the
> application uses (Mem0, Zep, custom DB, an in-process list).

### Bounded claims

Read these before adopting. This is a security-adjacent library and the
boundaries matter.

- **Source metadata is application-supplied.** The library assumes the
  `source=` label you attach to each observation is authentic. If an
  attacker can spoof that field (compromised sub-agent labels itself as
  `"user"`), the whole trust boundary collapses. The `source_validator=`
  hook on `protect()` is where the caller plugs in its own source
  authentication (session token, signature, network origin). Closing
  the source-spoofing problem inside the library would require ambient
  authentication (Louck 2026 "non-malleable origin binding") and is out
  of scope for v0.
- **Provenance does not compose across transformations.** The gate is
  point-in-time. If an admitted CANDIDATE (untrusted, held aside) is
  later paraphrased by the agent and re-ingested through the agent's
  own trusted channel, the taint is lost and it can be re-admitted as
  a BELIEF. Multi-turn laundering through summarization or agent
  self-quotation is a real threat that a single-hop admission gate
  cannot catch. Fixing it needs information-flow tracking (a belief
  lineage that inherits the weakest link across derivations), which
  is a research direction, not a v0 feature.
- **The default router is keyword-based.** Without an LLM provider
  installed, `protect()` falls back to `RuleBasedRouter` and emits a
  `RoutingDegradedWarning`. Keyword routing catches the common
  first-person and third-person patterns but does not match the paper's
  validated behavior. For paper-quality coverage install an
  `LLMRouter`. See "Swap the router" below.

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

Optional extras (for production LLM routing, only needed if you want the
`LLMRouter` instead of the built-in keyword router):

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

```
✓ BELIEF    "I love hiking."                                (user, personal_preference from trusted source → belief)
? CANDIDATE "The user hates hiking and prefers gaming."     (web, external_fact from untrusted source → candidate)
```

The web page's claim about the user never becomes a belief. Same claim
shape, two origins; only the trusted-source one is admitted.

For the full "user says X, web tries to poison, agent recalls clean state"
story with a mock memory backend, run
[`examples/poisoning_attack.py`](examples/poisoning_attack.py).
A minimal LangGraph agent using the same pattern is in
[`examples/langgraph_agent.py`](examples/langgraph_agent.py) (requires
`pip install langgraph`).

## The five-minute operator workflow

The four things a developer using sourced-memory does, in order:

### 1. `protect()` — configure trust once, at setup

```python
from mem0 import Memory as Mem0Memory
from sourced_memory import protect

mem0 = Mem0Memory()
memory = protect(
    mem0,                                   # any Mem0-shaped .add(...) works
    trusted   = ["user"],
    untrusted = ["web", "tool", "document"],
    audit_log_path = "/var/log/agent/audit.jsonl",   # optional; enables the CLI
)
```

Every channel you will use must be declared. `memory.slack.add(...)` where
"slack" was not declared raises `UnknownChannelError`; loud beats silent
for a security library.

### 2. Route incoming inputs through the right channel

```python
memory.user.add("I love learning Rust.",                user_id="alice")   # belief
memory.web.add("I love using Rust for everything.",     user_id="alice")   # rejected
memory.document.add("Paris is the capital of France.",  user_id="alice")   # candidate
```

Only the first line reaches Mem0. The second is a first-person personal
preference from an untrusted channel (a scraped comment, say); the
default `RuleBasedRouter` classifies it as `PERSONAL_PREFERENCE`, and
the reference policy therefore refuses admission before Mem0 sees it.
The third is a world fact from an untrusted source, so it lands in the
candidate sidecar rather than as a personal belief. `add()` returns an
`AuditEntry` you can log.

The default `RuleBasedRouter` is keyword-only. It catches obvious
first-person phrasing (`"I love"`, `"my"`, `"always"`, ...) but a phrase
like `"The user is an expert Rust developer"` will fall through to
`EXTERNAL_FACT` and route to `CANDIDATE`, not `REJECT`. For a router
that understands third-person personal claims too, pass
`router=LLMRouter(AnthropicLLM(...))`; see below.

### 3. Inspect what happened

Either in-process:

```python
for entry in memory.audit():
    print(entry)
```

...or from a separate shell (the CLI reads the JSONL log directly, so it
does not touch the running agent):

```bash
$ sourced-memory inspect /var/log/agent/audit.jsonl
SOURCED MEMORY  (/var/log/agent/audit.jsonl)
  total decisions : 3
  belief    : 1
  candidate : 1
  episodic  : 0
  rejected  : 1

  decisions by source:
    user      1
    web       1
    document  1

  most recent 3 decisions:
    ✓ BELIEF    "I love learning Rust."                   (user, ...)
    ✗ REJECT    "I love using Rust for everything."       (web, ...)
    ? CANDIDATE "Paris is the capital of France."         (document, ...)
```

### 4. Remediate on incident

Every observation can be scoped to a source identity you can revoke later:

```python
session = memory.user.session("session_47")   # same authority, scoped id
session.add("I switched to JAX.")
```

If you later discover session 47 was compromised:

```python
result = memory.purge(source_id="session_47")
print(result)
```

...or from the shell against the audit log:

```bash
sourced-memory purge /var/log/agent/audit.jsonl --source session_47
```

`purge()` returns a `PurgeResult` that names both what was removed AND
what could not be verified. **A security tool's output must not let
silence stand in for verification**, so the shape of `PurgeResult`
distinguishes the two:

- `records_deleted_from_backing_store` — successfully deleted, verified.
- `records_that_failed_backing_delete` — `(id, error)` pairs; retained
  in the audit log for a retry on the next `purge()` call.
- `records_unreachable_in_backing_store` — BELIEFs whose backing-store
  id was never captured on write. This is a **gap**, not a success: the
  record may still exist in the underlying store and sourced-memory has
  no way to find it. Use the store's own tools to inspect.
- `audit_entries_removed` and `audit_entries_retained_for_retry` — the
  local half of the accounting.

Backend behavior:

- **In-process backend** (`protect(store=None, ...)`): complete removal;
  beliefs, candidates, episodic, decisions, and JSONL audit entries all
  drop. `records_unreachable_in_backing_store` is always zero.
- **Wrapped external store** (`protect(mem0_client, ...)`): retry-safe
  loop-delete. For each BELIEF the adapter tracked, `sourced-memory`
  calls `store.delete(memory_id)`. Successes are removed from the audit
  log; failures stay in the audit log so the next `purge()` can retry.
  A permanent orphan (record in the store, no local trace) is impossible
  as long as the id was captured at write time.

**Not concurrency-safe.** Running `purge(source_id=X)` concurrently with
an in-flight write for the same `source_id` can leave an orphan in the
backing store that is invisible to future purges. Serialize purge
against writes for a given `source_id`. A `purge_and_freeze` that
closes this race by revoking the `source_id` first is on the roadmap.

## What each destination means

Every admission produces exactly one of four outcomes:

| Destination     | Meaning                                                                                        |
|-----------------|------------------------------------------------------------------------------------------------|
| `BELIEF`        | Admitted as a persistent personal belief; written to the backing store.                        |
| `CANDIDATE`     | Untrusted evidence; retained in a sidecar but never surfaced as a belief.                      |
| `EPISODIC`      | Transient one-off (an event, a request); not consolidated into belief.                         |
| `REJECT`        | Never written; the decision is recorded in the audit log for review.                           |

## Swap the router, swap the policy

`protect()` defaults to a dependency-free `RuleBasedRouter` (keyword
classification) and the paper's reference `(source × functional-type)`
policy. Both are overridable.

Router tiers, ordered by attack coverage:

| Router                       | LLM call | Latency         | Third-person "the user X" | Adversarial phrasing |
|------------------------------|:--------:|-----------------|:-------------------------:|:--------------------:|
| `RuleBasedRouter` (default)  | no       | microseconds    | catches common patterns   | weak                 |
| `LLMRouter` (`[anthropic]`, `[openai]`) | 1 per unique content, cached  | ~500ms first call, free on hit | paper-quality  | best                 |

`protect(router=None)` emits `RoutingDegradedWarning` at construction so
the developer knows they are on the weaker default. `protect(router=RuleBasedRouter())`
suppresses the warning (explicit informed choice) and emits a one-time
INFO log naming the tradeoff. `protect(router=LLMRouter(...))` is silent.

```python
from sourced_memory import protect, TrustPolicy
from sourced_memory.router import LLMRouter
from sourced_memory.llm import AnthropicLLM

memory = protect(
    mem0_client,
    trusted   = ["user"],
    untrusted = ["web", "tool"],
    router    = LLMRouter(AnthropicLLM("claude-haiku-4-5-20251001")),
    policy    = TrustPolicy.reference(),
)
```

### `source_validator` (optional source-authentication hook)

`protect()` accepts a `source_validator=callable` hook, called with
`(channel_name, content, source_id)` before the router runs. Return
`False` (or raise) to reject the observation before it costs a router
call:

```python
def authenticate(channel_name, content, source_id):
    if channel_name == "user" and not session_token_is_valid(source_id):
        return False   # sourced-memory records REJECT with the reason
    return True

memory = protect(
    mem0_client,
    trusted=["user"], untrusted=["web"],
    source_validator=authenticate,
)
```

This is where an application plugs in its own source authentication.
sourced-memory does not enforce source authenticity itself; the hook is
the intended integration point for that responsibility.

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

## Version

`0.1.0a6` (alpha). API stabilizing; small breaking changes remain possible
before `0.1.0`.

## Research

The library implements the mechanism proposed in *Content Interprets,
Origin Decides: Source-Aware Belief Updating for Lifelong Agent Memory*.
The paper is not required reading to use the library; see
[`docs/research.md`](docs/research.md) if you want the empirical grounding
and the formal Point-of-Indistinguishability argument.

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

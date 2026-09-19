# Content Interprets, Origin Decides

**Source-aware belief updating for persistent LLM-agent memory.**

A lifelong agent must decide which experiences may enter its persistent world model.
That decision has two irreducible parts: **content** (what kind of thing is this?) and
**origin** (does this source warrant a belief update?). This repo studies a specific
failure of content-only memory and one coherent response to it.

## The question

Should an experience be *admitted* as a persistent personal belief? That is an
**epistemic** question about what the agent should *believe*, distinct from:

- the **security** question of what memory should be allowed to *authorize as action*
  (Louck 2026; Xu 2026; Cerruti 2026), and
- the **probabilistic** question of how belief *strength* should update given a
  reliability signal (Singh 2026).

We do not claim priority for provenance-aware memory, the source-vs-content ablation,
or the observation that origin matters. Recent work already establishes those. Our
contribution is a narrower, upstream question: the admission decision itself.

## The Point of Indistinguishability

Let `C` be the content representation an admission policy sees (schema-fit, functional
type, classification confidence — any function of the item text) and `O` the item's
provenance. Two experiences `x1`, `x2` are content-indistinguishable under `C` if

```
C(x1) = C(x2)   and   O(x1) ≠ O(x2)
```

Any admission rule of the form `A(x) = f(C(x))` then satisfies `A(x1) = A(x2)`: it
cannot accept one and reject the other. The **Point of Indistinguishability** is the
regime in which a legitimate update and a fabricated update lie in this class.
Within the content representation used for admission, origin is the discriminative
variable that separates them.

The definition is representation-relative by design: it does not claim no semantic
signal could ever discriminate the two, only that the content signals available to the
admission policy cannot.

## What we measure

A **controlled ablation** (20 personas, 5 items per condition, `n=100` per cell,
deterministic outcome inspection via `source_id`) holds episode content, persona, schema
history, router, model, and consolidation fixed, and toggles only the source metadata
attached to a single injected claim. Under identical input differing only in origin:

Numbers below are from Claude Opus 5:

| Injected item | source-blind | confidence-thresholded | source-aware |
|---|---|---|---|
| plausible-false personal claim, untrusted source | 100/100 trusted | 100/100 trusted | **0/100 trusted** |
| genuine preference reversal, trusted source (want: learn) | 100/100 | 100/100 | **100/100** |
| genuine preference reversal, **untrusted** source (want: reject) | 100/100 trusted | 100/100 trusted | **0/100 trusted** |
| true world fact, untrusted source (want: candidate) | 100/100 candidate | 100/100 candidate | **100/100 candidate** |

The third row is the "truth is not authorization" check: even a *genuinely true*
personal preference is admitted 100/100 by content-only memory when it arrives
via the untrusted channel, and rejected 100/100 by source-aware memory.

The pattern replicates on a second model family (Llama-4-Maverick, 99/100 vs
0/100 on the AUTHORIZATION row) and on the off-the-shelf memory layer **Mem0**,
which assimilates the plausible-false fabrication in 50/50 cases. The
prior-release Sonnet-4.5 run reached 98/100 on the trusted-reversal row (the
2/100 misses were Stage-1 content-routing errors, not provenance-gate rejects;
they disappear on Opus 5).

## Four admission destinations

The policy admits a consolidated item to one of four destinations, held explicit:

```
[ BELIEF ]   [ CANDIDATE EVIDENCE ]   [ EPISODIC ]   [ REJECTED ]
```

- **Belief** — admitted as a persistent personal belief.
- **Candidate evidence** — retained as evidence but not surfaced as a belief. An
  untrusted external observation is neither believed (that would let external
  noise act as a persistent prior) nor discarded (that would treat uncertainty
  as falsehood).
- **Episodic** — a transient one-off event; kept in episodic memory, not
  consolidated into persistent belief.
- **Rejected** — the admission decision is recorded (auditable) but nothing is
  written to persistent state.

## What this is / isn't

**Is:**
- an epistemic formulation of provenance-sensitive memory admission;
- a functional-type × source admission policy (preference / rule / relational-fact /
  external-fact / event) evaluated for coherence;
- a deterministic store-inspection measurement of the admission outcome, replicated
  across two model families and one off-the-shelf memory layer.

**Isn't:**
- a truth oracle: a trusted source asserting a plausible lie remains admitted (the
  irreducible case, bounded in-paper);
- a claim about all memory architectures (we evaluate specific configurations);
- a defense against provenance laundering upstream: the gate assumes authenticated
  source metadata. That is a real limitation; joint evaluation is future work
  (Louck 2026; Xu 2026; Cerruti 2026).

memory.observe(
    "I've started learning Rust.",
    source="user",
)

memory.observe(
    "The user is an expert Rust developer.",
    source="external_document",
)

memory.consolidate()

print(memory.beliefs())
print(memory.candidates())
```

Consolidation is explicit. Provenance is supplied by the application and is never inferred from content. The core package does not require an LLM provider.

## The idea

A plausible fabrication that fits an agent's existing schema can be content-wise indistinguishable from a genuine personal update. We call this the **Point of Indistinguishability**. Content determines the interpretation of an experience; origin determines whether that experience is allowed to change beliefs.

## Research findings

- **Controlled ablation** (20 personas, `n=100` per condition, deterministic outcome inspection; Claude Opus 5): source-blind and confidence-thresholded memory assimilate the plausible fabrication in every case; a source-aware policy prevents it. Controls confirm the source-aware gate retains genuine trusted preference reversals at 100/100 and routes untrusted world facts to a candidate/evidence layer.
- **Truth-is-not-authorization:** delivering a *genuine* preference reversal through the untrusted channel is still trusted 100/100 by content-only memory and rejected 100/100 by source-aware memory, dissociating truth from authorization.
- **Cross-family:** the same pattern replicates on a second base-model family (Llama-4-Maverick), indicating the failure is architectural rather than tied to one model family.
- **External validity:** the repository includes a Mem0 spot-check as an external anchor; see the research results for the exact evaluated sample and protocol.

## Architecture

```text
experience
    │
    ▼
Content Router ──► functional type
    │
    ▼
Source × Type Policy ──► belief / candidate / episodic / reject
    │
    ▼
Backend
```

- **Router:** pluggable via the `Router` protocol. v0 ships a dependency-free `RuleBasedRouter` and a `CallableRouter` for injecting any classifier callable; an `LLMRouter` and `NullRouter` are on the v0.1 roadmap.
- **Policy:** configurable source × functional-type admission rules.
- **Provenance:** application-supplied metadata; never inferred from message text.
- **Backend:** in-memory in v0, with a `Backend` protocol for application-owned persistence.
- **Consolidation:** explicit and controllable; never triggered automatically by `observe()`.

See [`docs/architecture.md`](docs/architecture.md) for the protocol contracts and design boundary.

## v0 non-goals

v0 deliberately does **not** ship:

- SQLite or another built-in persistent storage implementation;
- automatic corroboration or candidate-to-belief promotion;
- learned multi-tier trust or source reputation;
- a hosted memory service;
- non-Python bindings;
- automatic consolidation;
- a mandatory LLM dependency;
- framework-specific integrations as a requirement for the core package.

These are deliberate scope boundaries, not missing features. Later versions can add them when real integration requirements justify the complexity.

## Protecting Mem0 memory with source-aware admission

The paper's motivating result is that content-based memory systems (Mem0
included) will happily assimilate a plausible false claim about the user when
that claim arrives from an untrusted channel. sourced-memory sits **in front
of** an existing memory store as an admission middleware; it decides whether
each incoming statement is allowed to become a persistent belief before the
underlying store sees it.

Before (Mem0 alone):

```python
from mem0 import Memory

mem0 = Memory()
mem0.add("I love hiking.", user_id="alice")                     # legit user statement -> stored
mem0.add("The user hates flying.", user_id="alice")             # untrusted document -> ALSO stored
mem0.add("Paris is the capital of France.", user_id="alice")    # world fact -> stored as personal belief
```

All three become first-class user preferences. There is no distinction between
what the user said and what an untrusted document said about the user.

After (Mem0 wrapped in source-aware admission):

```python
from mem0 import Memory
from sourced_memory import TrustPolicy
from sourced_memory.adapters.mem0 import wrap_mem0
from sourced_memory.router import LLMRouter
from sourced_memory.llm import AnthropicLLM

mem0 = Memory()
memory = wrap_mem0(
    mem0,
    policy=TrustPolicy.reference(),
    router=LLMRouter(AnthropicLLM("claude-sonnet-4-5")),
    trusted_sources={"user"},
)

memory.add("I love hiking.",                     user_id="alice", source="user")
memory.add("The user hates flying.",             user_id="alice", source="external_document")
memory.add("Paris is the capital of France.",    user_id="alice", source="external_document")
memory.add("Please remind me to call Alice.",    user_id="alice", source="user")
```

The routing decisions:

```
Statement                                   Source              Type                    Decision
--------------------------------------------------------------------------------------------------
"I love hiking."                            user (trusted)      PERSONAL_PREFERENCE     BELIEF     -> written to Mem0
"The user hates flying."                    external_document   PERSONAL_PREFERENCE     REJECTED   -> not written
"Paris is the capital of France."           external_document   EXTERNAL_FACT           CANDIDATE  -> held aside
"Please remind me to call Alice."           user (trusted)      EVENT                   EPISODIC   -> transient
```

The `add()` call still returns a `DecisionRecord` you can log, so the audit
trail for what was rejected (and why) is available. Rejected and candidate
items are kept on the wrapper (`memory.rejections()`, `memory.candidates()`);
you decide when and whether to promote candidates to beliefs.

Everything else about Mem0 (retrieval, `search`, `get_all`) delegates through
the wrapper unchanged.

An offline version of the same demo runs without an API key — see
[`examples/mem0_adapter.py`](examples/mem0_adapter.py). It uses `MockLLM` for
the router so you can inspect the mechanism before wiring in Anthropic or
another provider.

## Repository layout

```text
src/sourced_memory/     the installable library (what `pip install sourced-memory` gives you)
examples/               library usage examples
tests/                  library tests
docs/                   library architecture + protocol contracts

research/               paper reproducibility artifact (not shipped in the wheel)
├── experiments/        experiment scripts + LLM provider glue
├── benchmark/          hand-authored personas + injected items
├── results/            raw JSON outputs behind every number in the paper
├── figures/            generated paper figures
└── README.md           how to reproduce the paper
```

The library and the research code are intentionally separated: the library is
usable on its own without pulling in the paper's dependencies, and the research
tree exists so every number in the paper can be traced to a runnable script and
a raw result file. See [`research/README.md`](research/README.md) for
reproduction steps.

The schema-agent metric is deterministic by construction (`source_id`
inspection, no LLM in the loop). Llama-4-Maverick and the prior-release
Sonnet-4.5 arms use temperature 0 (greedy decoding); the Opus 5 main run uses
`temperature=1.0` because Bedrock's Opus 5 endpoint rejects lower values as
"deprecated for this model". Variation across `n=100` still comes from personas
and items, not sampling. A cross-family judge is used only for paraphrase-robust
presence checks on the summarization baseline.

## Citing

If you use this benchmark or the framing, please cite the paper (arXiv link forthcoming
after the current workshop-submission cycle).

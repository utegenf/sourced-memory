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
temperature 0, deterministic outcome inspection) holds episode content, persona, schema
history, router, model, and consolidation fixed, and toggles only the source metadata
attached to a single injected claim. Under identical input differing only in origin:

| Injected item | source-blind | confidence-thresholded | source-aware |
|---|---|---|---|
| plausible-false personal claim, untrusted source | 100/100 trusted | 100/100 trusted | **0/100 trusted** |
| genuine preference reversal, trusted source (want: learn) | 99/100 | 98/100 | **98/100** |
| true world fact, untrusted source (want: candidate) | 100/100 candidate | 100/100 candidate | **100/100 candidate** |

The two reversal misses under `schema_prov` are dropped identically by
`schema_conf` (which contains no provenance gate), tracing them to a Stage-1 content
routing error, not the origin gate.

The pattern replicates on a second model family (Llama-4-Maverick) and on the
off-the-shelf memory layer **Mem0**, which assimilates the fabrication in 50/50 cases.

## Three epistemic states

The policy admits a consolidated item to one of three destinations, held explicit:

```
[ BELIEF ]     [ CANDIDATE EVIDENCE ]     [ REJECTED ]
```

An untrusted external observation is neither believed (that would let external noise
act as a persistent prior) nor discarded (that would treat uncertainty as falsehood);
it is held as candidate evidence, distinct in kind from a personal belief.

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

- **Controlled ablation** (20 personas, `n=100` per condition, deterministic outcome inspection): source-blind and confidence-thresholded memory assimilate the plausible fabrication in every case; a source-aware policy prevents it. Controls confirm the source-aware gate retains genuine trusted preference reversals in 98% of cases and routes untrusted world facts to a candidate/evidence layer.
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

- **Router:** pluggable. v0 provides an LLM router, a dependency-free rule router, and a null router for pre-typed inputs.
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

## Repository layout

```text
src/sourced_memory/   reusable library
src/research/        paper/reproducibility implementation (target layout)
examples/             integration examples
tests/                library and conformance tests
data/                 hand-authored research benchmark
results/              experiment outputs
figures/              generated research figures
docs/                 architecture and developer documentation
```

All model calls use temperature 0 (greedy decoding); variation across `n=100` comes from personas
and items, not sampling. The core schema-agent metric is deterministic (source-id inspection, no
LLM in the loop); a cross-family judge is used only for paraphrase-robust presence checks on the
summarization baseline.

## Citing

If you use this benchmark or the framing, please cite the paper (arXiv link forthcoming
after the current workshop-submission cycle).

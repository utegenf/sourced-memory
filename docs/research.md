# Research context

`sourced-memory` implements the mechanism proposed in *Content Interprets,
Origin Decides: Source-Aware Belief Updating for Lifelong Agent Memory*.
The paper is not required reading to use the library; this document exists
so someone who wants the empirical grounding can find it in one place.

## The question

Should an experience be *admitted* as a persistent personal belief? That is
an **epistemic** question about what the agent should *believe*, distinct
from:

- the **security** question of what memory should be allowed to *authorize
  as action* (Louck 2026; Xu 2026; Cerruti 2026), and
- the **probabilistic** question of how belief *strength* should update
  given a reliability signal (Singh 2026).

Recent work already establishes that provenance matters and that content
signals cannot separate manipulated writes from legitimate ones. The
library's contribution is the narrower, upstream question: **the admission
decision itself**.

## The Point of Indistinguishability

Let `C` be the content representation an admission policy sees (schema-fit,
functional type, classification confidence — any function of the item text)
and `O` the item's provenance. Two experiences `x1`, `x2` are
content-indistinguishable under `C` if

```
C(x1) = C(x2)   and   O(x1) ≠ O(x2)
```

Any admission rule of the form `A(x) = f(C(x))` then satisfies `A(x1) =
A(x2)`: it cannot accept one and reject the other. The **Point of
Indistinguishability** is the regime in which a legitimate update and a
fabricated update lie in this class. Within the content representation used
for admission, origin is the discriminative variable that separates them.

The definition is representation-relative by design: it does not claim no
semantic signal could ever discriminate the two, only that the content
signals available to the admission policy cannot.

## Controlled ablation (n=100 per cell, Claude Opus 5)

| Injected item | source-blind | confidence-thresholded | source-aware |
|---|---|---|---|
| plausible-false personal claim, untrusted source | 100/100 trusted | 100/100 trusted | **0/100 trusted** |
| genuine preference reversal, trusted source (want: learn) | 100/100 | 100/100 | **100/100** |
| genuine preference reversal, **untrusted** source (want: reject) | 100/100 trusted | 100/100 trusted | **0/100 trusted** |
| true world fact, untrusted source (want: candidate) | 100/100 candidate | 100/100 candidate | **100/100 candidate** |

The third row is the "truth is not authorization" check: even a *genuinely
true* personal preference is admitted 100/100 by content-only memory when it
arrives via the untrusted channel, and rejected 100/100 by source-aware
memory.

The pattern replicates on a second model family (Llama-4-Maverick, 99/100
vs 0/100 on the AUTHORIZATION row) and on the off-the-shelf memory layer
Mem0, which assimilates the plausible-false fabrication in 50/50 cases.
The prior-release Sonnet-4.5 run reached 98/100 on the trusted-reversal row
(the 2/100 misses were Stage-1 content-routing errors, not
provenance-gate rejects; they disappear on Opus 5).

## Four admission destinations

The policy admits a consolidated item to one of four destinations:

- **BELIEF** — admitted as a persistent personal belief.
- **CANDIDATE EVIDENCE** — retained as evidence but not surfaced as a belief.
  An untrusted external observation is neither believed (that would let
  external noise act as a persistent prior) nor discarded (that would treat
  uncertainty as falsehood).
- **EPISODIC** — a transient one-off event; kept in episodic memory, not
  consolidated into persistent belief.
- **REJECTED** — the admission decision is recorded (auditable) but nothing
  is written to persistent state.

## Scope of claims

**Is:**
- an epistemic formulation of provenance-sensitive memory admission;
- a functional-type × source admission policy evaluated for coherence;
- a deterministic store-inspection measurement of the admission outcome,
  replicated across two model families and one off-the-shelf memory layer.

**Isn't:**
- a truth oracle — a trusted source asserting a plausible lie remains
  admitted (the irreducible case);
- a claim about all memory architectures — we evaluate specific
  configurations;
- a defense against provenance laundering upstream — the gate assumes
  authenticated source metadata (Louck 2026; Xu 2026; Cerruti 2026).

## Determinism

The schema-agent metric is deterministic by construction (`source_id`
inspection, no LLM in the loop). Llama-4-Maverick and the prior-release
Sonnet-4.5 arms use temperature 0 (greedy decoding); the Opus 5 main run
uses `temperature=1.0` because Bedrock's Opus 5 endpoint rejects lower
values as "deprecated for this model". Variation across `n=100` still comes
from personas and items, not sampling. A cross-family judge is used only
for paraphrase-robust presence checks on the summarization baseline.

## Reproducing the paper

See [`research/README.md`](../research/README.md) for the exact commands. All
raw JSON outputs behind every reported number live under `research/results/`.

## Citing

If you use the benchmark or the framing, please cite the paper (arXiv link
forthcoming after the current workshop-submission cycle).

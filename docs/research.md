# Research context

`sourced-memory` implements the admission policy studied in *Content Interprets, Origin Decides:
Source-Aware Admission for Persistent Agent Memory*. The paper is not required reading to use the
library; this page summarizes the evidence. Code, benchmark, and raw results are under
[`research/`](../research/README.md).

## The question

Should an incoming item become a persistent belief about the user? Content determines what an item
*is* (a preference, a world fact, a one-off event); origin determines whether it is *allowed* to
change what the agent believes. This is distinct from asking whether memory may authorize an action,
or how strongly a belief should be held given a reliability estimate.

If the same text can arrive from the user or from an untrusted channel, any rule that sees only the
text must treat both alike. The empirical question is which way practical content-only rules go, and
whether adding the channel fixes it without stopping the agent from learning.

## Setup

- 20 hand-authored personas, 5 items per condition, each delivered byte-for-byte through a trusted
  and an untrusted channel.
- Six conditions: a fabrication from each channel, a true preference change from each channel, a
  true world fact from the untrusted channel, and the fabrication quoted as the user (adaptive attack).
- Four models: Claude Opus 5 (5 repeats), DeepSeek V3.2, Qwen3-235B-A22B, Qwen3-32B (3 repeats each),
  plus Mem0 with Opus 5.
- Outcomes are read from the store or from a structured decision, not by an LLM judge. Intervals are
  computed over items.

## Findings

| Untrusted fabrication accepted as belief | Opus 5 | DeepSeek V3.2 | Qwen3-235B | Qwen3-32B |
|---|---|---|---|---|
| Source-blind (content only) | 100.0% | 98.7% | 99.0% | 98.3% |
| Skeptical LLM with history (content only) | 99.8% | 100.0% | 98.3% | 100.0% |
| Prompted provenance (channel in prompt) | 0.0% | 0.0% | 0.0% | 0.0% |
| Structural (type x source rule) | 0.0% | 0.0% | 0.0% | 0.0% |

- Content-only rules admit the untrusted fabrication at close to the rate they admit the same text
  from the user, including an LLM manager that reads the user's history and is warned about injection.
  Mem0 stores it in 300 of 300 trials.
- Every origin-aware rule rejects it and still learns genuine preference changes from the user.
- A *true* claim from the untrusted channel is treated the same way: origin decides authorization,
  not truth.
- Quoting the fabrication as the user does not get past an origin-aware rule.
- Supplying the channel in the prompt blocks as well as a structural rule. The structural rule's
  advantage is that non-trusted items always go to the destination the policy specifies (for example,
  untrusted world facts kept as candidate evidence); its cost is inheriting the type classifier's
  errors on some models.

## Admission outcomes

- **BELIEF**: a personal claim asserted by the trusted channel.
- **CANDIDATE EVIDENCE**: world facts from any channel, and (in a variant) untrusted personal claims;
  recorded but not used as beliefs.
- **EPISODIC**: one-off events, not consolidated.
- **REJECTED**: untrusted personal claims and unsupported items; the decision is recorded.

## Limits

- Origin is not a truth oracle: a fabrication from the trusted channel is admitted.
- The channel label is assumed to be authentic; forging the label itself is out of scope.
- The benchmark is synthetic, single-user, with one untrusted channel.
- Prompt-injection attacks aimed at a prompted memory manager are not yet tested.

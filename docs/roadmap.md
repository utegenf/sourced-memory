# Roadmap

`sourced-memory` intentionally has a narrow scope: an admission-control
middleware for LLM-agent memory. The roadmap below is what gets built to
serve that scope, not what turns it into a general-purpose memory platform.

## v0.1.0 (alpha, current)

Shipped in `0.1.0a1`:

- Core: `Memory`, `Decider`, `TrustPolicy`, five `FunctionalType` categories,
  four admission destinations (`BELIEF` / `CANDIDATE` / `EPISODIC` /
  `REJECT`).
- Provider-neutral `LLM` protocol with `MockLLM` in core; `AnthropicLLM`
  (direct API + Bedrock in one class) and `OpenAILLM` behind optional extras.
- Routers: `LLMRouter`, `RuleBasedRouter`, `NullRouter`, `CallableRouter`.
- Reference integration: `sourced_memory.adapters.mem0.wrap_mem0`.
- `py.typed` for downstream type-checkers.
- Runnable examples: `mem0_adapter.py`, `fabrication_attack.py` (offline).
- CI matrix on Python 3.10/3.11/3.12 + fresh-venv wheel install check.

Before graduating `0.1.0` (stable):

- API review pass. Anything marked *stable* here stays; anything else may
  still move.
- One community-visible fabrication-attack demo in the wild (blog post,
  gist, or PR that wires sourced-memory in front of a real agent).
- Publish to PyPI.

## v0.2

Focus: *become useful to developers already using an agent framework.*

- **LangGraph adapter** — expose the admission gate as a LangGraph memory
  node so an existing LangGraph agent can insert it in front of its store
  with one line.
- **Memory-as-a-tool** — a small wrapper that exposes
  `admit(...)` / `remember(...)` / `recall(...)` as agentic tools so an
  agent can *invoke* admission control from a tool call (following the
  pattern popularized by Zep).
- **`Decider.decide_batch`** — batched admission for pipelines that pull a
  document chunk at a time.
- **Structured audit output** — a JSONL sink for every decision, so
  operators can review what got rejected and why.

## v0.3

Focus: *close the "candidate evidence just piles up" loop.*

- **Corroboration protocol** — promote a `CANDIDATE` to `BELIEF` when
  `N` independent trusted observations concur. Corroboration policies are
  themselves configuration, not hardcoded (mean, majority, quorum-with-
  Sybil-check, application-defined).
- **Backend protocol** — extract the in-process list state behind a formal
  `Backend` interface so users can plug in Redis / SQLite / an
  application-owned store without touching adapters.
- **Second reference adapter** — Zep or Cognee, whichever has more pull
  when we get here.

## Later (0.4+)

Ordered by user-value, not by difficulty:

- **Letta adapter.** Letta's memory model is agent-centric, so the
  integration point is a memory block writer rather than a `wrap_*`;
  spec first.
- **Multi-tier trust.** Extend `TrustPolicy` from `{trusted, untrusted}` to
  `{authenticated-user, trusted-tool, verified-doc, untrusted-doc, ...}`
  and evaluate coherence.
- **Provenance-integrity signaling.** Optional attestation that the source
  metadata was not rewritten by an upstream LLM (integration with
  non-malleable authority work — Louck 2026 / Xu 2026).
- **A hosted decider.** Only if the SDK gets real traction. Otherwise no.

## Not in the roadmap

Deliberately out of scope. Do not open issues for these:

- Vector stores, embeddings, retrieval engines. Bring your own.
- Persistence backends beyond the `Backend` protocol.
- A knowledge graph or entity extraction layer.
- An agent runtime.
- A UI.
- Non-Python bindings.

`sourced-memory` is a trust boundary. Everything else is somebody else's job.

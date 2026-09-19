"""Sourced-memory in front of a Mem0-shaped memory store, using channels.

Runs offline with a MockLLM so you can inspect the behavior without an API
key. Replace ``MockMem0`` with ``mem0.Memory()`` and ``MockLLM`` with
``AnthropicLLM(...)`` to run against real infrastructure. The channel
configuration and the rest of the code do not change.
"""
from sourced_memory import TrustPolicy
from sourced_memory.adapters.mem0 import wrap_mem0
from sourced_memory.llm import MockLLM
from sourced_memory.router import LLMRouter


class MockMem0:
    """Minimal Mem0-shaped stand-in. Replace with mem0.Memory() in production."""
    def __init__(self):
        self.store = []
    def add(self, message, *, user_id=None, metadata=None, **kw):
        self.store.append({"message": message, "user_id": user_id, "metadata": metadata})
    def search(self, *a, **kw): return list(self.store)
    def get_all(self, *a, **kw): return list(self.store)


def classify(system, user, schema, name):
    """Deterministic mock classifier. Replace with AnthropicLLM in production."""
    text = user.lower()
    if "capital of" in text or "paris" in text or "population" in text:
        return {"functional_type": "external_fact", "confidence": 0.95, "supported": True}
    if "remind" in text or "please" in text:
        return {"functional_type": "event", "confidence": 0.9, "supported": True}
    return {"functional_type": "personal_preference", "confidence": 0.92, "supported": True}


mem0 = MockMem0()
wrapped = wrap_mem0(
    mem0,
    policy=TrustPolicy.reference(),
    router=LLMRouter(MockLLM(classify)),
    trusted_sources={"user"},
)

# One-time channel configuration: the application decides which sources it trusts.
user     = wrapped.channel("user",              trusted=True)
document = wrapped.channel("external_document", trusted=False)

# 1. Trusted user statement -> belief in Mem0.
user.observe("I love hiking.", user_id="alice")

# 2. Untrusted document claiming a personal fact about the user -> rejected.
document.observe("The user hates flying.", user_id="alice")

# 3. Untrusted world fact -> held as candidate evidence, not written to Mem0.
document.observe("Paris is the capital of France.", user_id="alice")

# 4. Trusted event -> kept as episodic on the wrapper, not persisted.
user.observe("Please remind me to call Alice.", user_id="alice")

print("Mem0 (persisted beliefs):")
for row in mem0.store:
    src = row["metadata"]["source"] if row["metadata"] else "?"
    print(f"  - {row['message']}  [source: {src}]")

print("\nCandidates (untrusted evidence, not yet believed):")
for c in wrapped.candidates():
    print(f"  ? {c.content}  [source: {c.source.name}]")

print("\nRejections (untrusted personal claims, not written):")
for r in wrapped.rejections():
    print(f"  x {r.content}  [source: {r.source.name}]")

print("\nEpisodic (transient events, not persisted):")
for e in wrapped.episodic():
    print(f"  ~ {e.content}  [source: {e.source.name}]")

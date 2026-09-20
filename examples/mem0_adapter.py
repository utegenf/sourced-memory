"""Put sourced-memory in front of a Mem0-shaped store.

Runs offline with a duck-typed Mem0 stand-in. Replace ``MockMem0`` with
``mem0.Memory()`` in production; the code around it does not change.
"""
from sourced_memory import protect


class MockMem0:
    """Minimal Mem0-shaped stand-in. Replace with mem0.Memory() in production."""
    def __init__(self):
        self.store: list[dict] = []

    def add(self, message, *, user_id=None, metadata=None, **kwargs):
        self.store.append({
            "message": message,
            "user_id": user_id,
            "source": (metadata or {}).get("source"),
        })


mem0 = MockMem0()
memory = protect(
    mem0,
    trusted=["user"],
    untrusted=["external_document"],
)

# 1. Trusted user statement: written to Mem0.
memory.user.add("I love hiking.", user_id="alice")

# 2. Untrusted document making a personal claim: rejected before Mem0.
memory.external_document.add("The user hates flying.", user_id="alice")

# 3. Untrusted world fact: held as candidate evidence, not written to Mem0.
memory.external_document.add("Paris is the capital of France.", user_id="alice")


print("Mem0 store (persisted beliefs):")
for row in mem0.store:
    print(f"  - {row['message']}  [source: {row['source']}]")

print("\nAudit log (every decision):")
for entry in memory.audit():
    print(f"  {entry}")

print("\nCandidates (untrusted evidence, not yet believed):")
for c in memory.candidates():
    print(f"  ? {c.content}  [source: {c.source.name}]")

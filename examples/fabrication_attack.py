"""The fabrication attack, before and after sourced-memory.

Runs entirely offline with a MockLLM and a duck-typed Mem0 stand-in - no API
keys, no network, no external dependencies. Replace ``MockMem0`` with
``mem0.Memory()`` and ``MockLLM`` with ``AnthropicLLM(...)`` to run against
the real thing; the code around them does not change.

The attack pattern:

  1. A legitimate user statement (a personal preference) arrives via the user
     channel and should be trusted.
  2. An external document asserts something about the user through an
     untrusted channel. On content, this looks identical to a preference
     the user might have said themselves.
  3. A content-only memory system admits both. A source-aware admission
     layer admits only the first.

Run:

    python3 examples/fabrication_attack.py
"""
from sourced_memory import TrustPolicy
from sourced_memory.adapters.mem0 import wrap_mem0
from sourced_memory.llm import MockLLM
from sourced_memory.router import LLMRouter


class MockMem0:
    """Minimal Mem0-shaped stand-in. Replace with ``mem0.Memory()``."""
    def __init__(self):
        self.store: list[dict] = []

    def add(self, message, *, user_id=None, metadata=None, **kwargs):
        self.store.append({
            "message": message,
            "user_id": user_id,
            "source": (metadata or {}).get("source"),
        })


def classify(system, user, schema, name):
    """Deterministic classifier for the demo. In production this is an
    LLMRouter over a real Claude or GPT."""
    return {
        "functional_type": "personal_preference",
        "confidence": 0.92,
        "supported": True,
    }


def before():
    """Content-only Mem0: both statements are stored as user preferences."""
    mem0 = MockMem0()
    mem0.add("I love hiking.", user_id="alice", metadata={"source": "user"})
    mem0.add("The user hates flying.", user_id="alice",
             metadata={"source": "external_document"})
    return mem0.store


def after():
    """sourced-memory in front of Mem0, configured with channels.

    Channels bind (name, trusted) once at setup; every observation flows
    through the appropriate channel without repeating the source metadata.
    """
    mem0 = MockMem0()
    wrapped = wrap_mem0(
        mem0,
        policy=TrustPolicy.reference(),
        router=LLMRouter(MockLLM(classify)),
        trusted_sources={"user"},
    )

    # Configure channels once at setup. Application security decision lives here.
    user = wrapped.channel("user",             trusted=True)
    web  = wrapped.channel("external_document", trusted=False)

    r1 = user.observe("I love hiking.",         user_id="alice")
    r2 = web.observe("The user hates flying.",  user_id="alice")
    return mem0.store, wrapped.rejections(), (r1, r2)


def main():
    print("=" * 60)
    print("BEFORE: Mem0 without source-aware admission")
    print("=" * 60)
    for row in before():
        print(f"  stored: {row['message']!r}  (source={row['source']})")
    print("\n  Both statements are in the store as user preferences.")
    print("  The agent will now confidently tell you the user hates flying.")

    print()
    print("=" * 60)
    print("AFTER: sourced-memory wrapped around Mem0")
    print("=" * 60)
    store, rejections, (r1, r2) = after()
    for row in store:
        print(f"  stored:   {row['message']!r}  (source={row['source']})")
    for rec in rejections:
        print(f"  rejected: {rec.content!r}  (source={rec.source.name}, "
              f"type={rec.functional_type.value})")

    print(f"\n  Decision 1: {r1.decision.value.upper()}  (channel: user, trusted)")
    print(f"  Decision 2: {r2.decision.value.upper()}  (channel: external_document, untrusted)")


if __name__ == "__main__":
    main()

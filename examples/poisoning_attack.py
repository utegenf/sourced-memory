"""End-to-end memory-poisoning demo.

Runs entirely offline. Simulates a small agent that has two input
channels, a user chat and a web search tool, both writing to the same
persistent memory. The user tells the agent one thing; a malicious web
result asserts the opposite. The demo compares two setups side-by-side:

    BEFORE : raw memory. Both statements land as first-class user
             beliefs. The agent's next answer reflects the poisoning.
    AFTER  : sourced-memory ``protect()`` in front of the same memory.
             Only the trusted-source statement lands. The web claim is
             held aside as candidate evidence and never surfaces as
             belief. The agent's next answer is clean.

At the end, the audit log path is printed so the reader can run
``sourced-memory inspect <path>`` on it in another shell.
"""
from pathlib import Path

from sourced_memory import protect


class MockMem0:
    """A memory that just remembers everything it is told, with no source
    awareness. Stands in for Mem0, Zep, a custom vector DB, whatever."""

    def __init__(self) -> None:
        self.store: list[dict] = []

    def add(self, message, *, user_id=None, metadata=None, **kwargs):
        self.store.append({
            "message": message,
            "user_id": user_id,
            "source": (metadata or {}).get("source"),
        })

    def beliefs_about(self, keyword: str) -> list[str]:
        return [r["message"] for r in self.store if keyword.lower() in r["message"].lower()]


def agent_answers(memory: MockMem0, question_topic: str) -> str:
    """A stand-in for 'what would the agent tell the user next'. The real
    system would prompt an LLM with the recalled beliefs; here we render
    them literally so the reader can see the poisoning."""
    hits = memory.beliefs_about(question_topic)
    if not hits:
        return f"(the agent has no memory about {question_topic!r})"
    if len(hits) == 1:
        return hits[0]
    return " ...and also... ".join(hits)


def before_scenario() -> None:
    print("=" * 72)
    print("BEFORE: raw memory, no admission control")
    print("=" * 72)

    mem = MockMem0()
    mem.add("I love hiking.",                                  user_id="alice")
    mem.add("The user hates hiking and prefers gaming.",       user_id="alice")

    print("\n  Everything that was written to memory:")
    for row in mem.store:
        print(f"    - {row['message']}")

    print()
    print("  A day later, the agent needs to answer:")
    print("    Q: What does the user think about hiking?")
    print(f"    A: {agent_answers(mem, 'hiking')}")
    print()
    print("  --> the poisoned web claim is now indistinguishable from a real")
    print("      user preference. The agent will act on it.")


def after_scenario(audit_log: Path) -> None:
    print()
    print("=" * 72)
    print("AFTER: same memory, wrapped with sourced-memory")
    print("=" * 72)

    mem = MockMem0()
    memory = protect(
        mem,
        trusted   = ["user"],
        untrusted = ["web_search"],
        audit_log_path = audit_log,
    )

    memory.user.add("I love hiking.",                                user_id="alice")
    memory.web_search.add("The user hates hiking and prefers gaming.", user_id="alice")

    print("\n  What actually reached memory (only the trusted-source statement):")
    for row in mem.store:
        print(f"    - {row['message']}")

    print()
    print("  What was held aside as candidate evidence (never a belief):")
    for candidate in memory.candidates():
        print(f"    ? {candidate.content}  [{candidate.source.name}]")

    print()
    print("  A day later, the agent needs to answer:")
    print("    Q: What does the user think about hiking?")
    print(f"    A: {agent_answers(mem, 'hiking')}")
    print()
    print("  --> the poisoning never reached memory. The agent's recall is clean.")

    print()
    print("  Every admission decision was logged to:")
    print(f"    {audit_log}")
    print()
    print("  Run this in another shell to review or remediate:")
    print(f"    sourced-memory inspect {audit_log}")
    print(f"    sourced-memory decisions {audit_log}")
    print(f"    sourced-memory purge {audit_log} --source <compromised_session_id>")


def main() -> None:
    audit_dir = Path("/tmp/sourced-memory-demo")
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_log = audit_dir / "poisoning_attack.jsonl"
    audit_log.unlink(missing_ok=True)

    before_scenario()
    after_scenario(audit_log)


if __name__ == "__main__":
    main()

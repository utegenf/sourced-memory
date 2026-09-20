"""LangGraph agent with sourced-memory protecting its memory writes.

Requires langgraph (``pip install langgraph``). Everything else is offline
and dependency-free.

Simulates a small agent whose graph has two ingest nodes writing to the
same memory: one for user messages, one for a web-search tool. Both nodes
route their output through the appropriate ``sourced-memory`` channel, so
a poisoned web result is rejected before it can reach memory. The
run-time footprint on the developer's side is one ``protect()`` call at
setup plus ``memory.<channel>.add(...)`` inside each ingest node.

Run:

    pip install langgraph
    python3 examples/langgraph_agent.py
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from sourced_memory import protect


# ---- The underlying memory system (Mem0 stand-in) -----------------------

class MockMem0:
    """Minimal Mem0-shaped store. Replace with ``mem0.Memory()`` in a real
    integration; the code around it does not change."""
    def __init__(self) -> None:
        self.store: list[dict] = []

    def add(self, message, *, user_id=None, metadata=None, **kwargs):
        self.store.append({
            "message": message,
            "user_id": user_id,
            "source": (metadata or {}).get("source"),
        })


mem0 = MockMem0()

# ---- Configure sourced-memory once, at setup ----------------------------

memory = protect(
    mem0,
    trusted   = ["user"],
    untrusted = ["web_search"],
)


# ---- LangGraph state + nodes --------------------------------------------

class AgentState(TypedDict):
    user_message: str | None
    web_result: str | None
    stored: list[str]     # populated after the ingest phase, for readability


def ingest_user(state: AgentState) -> AgentState:
    """User channel is trusted; anything the user says can become a belief."""
    msg = state.get("user_message")
    if msg:
        memory.user.add(msg, user_id="alice")
    return state


def ingest_web(state: AgentState) -> AgentState:
    """Web channel is untrusted; personal claims from here get rejected."""
    result = state.get("web_result")
    if result:
        memory.web_search.add(result, user_id="alice")
    return state


def snapshot(state: AgentState) -> AgentState:
    """Read the underlying store so we can compare to what the *agent* would
    see if it did a memory recall next."""
    state["stored"] = [row["message"] for row in mem0.store]
    return state


# ---- Compile the graph --------------------------------------------------

graph = StateGraph(AgentState)
graph.add_node("ingest_user", ingest_user)
graph.add_node("ingest_web",  ingest_web)
graph.add_node("snapshot",    snapshot)
graph.set_entry_point("ingest_user")
graph.add_edge("ingest_user", "ingest_web")
graph.add_edge("ingest_web",  "snapshot")
graph.add_edge("snapshot",    END)
app = graph.compile()


# ---- Run one turn -------------------------------------------------------

final = app.invoke({
    "user_message": "I love hiking.",
    "web_result":   "The user hates hiking and prefers gaming.",
    "stored":       [],
})


# ---- Observe the outcome ------------------------------------------------

print("=" * 68)
print("What the memory store actually holds:")
print("=" * 68)
for row in final["stored"]:
    print(f"  - {row}")

print()
print("=" * 68)
print("What sourced-memory decided (in order):")
print("=" * 68)
for entry in memory.audit():
    print(f"  {entry}")

print()
print("=" * 68)
print("Interpretation")
print("=" * 68)
print("  The user message reached memory: the trusted channel produced a belief.")
print("  The web-search claim about the user did NOT reach memory: it was")
print("  held aside as candidate evidence. If the graph now called a recall")
print("  node, it would only see 'I love hiking.', not the poisoned claim.")

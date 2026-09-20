"""Tests for the protect() facade and its supporting types."""
import pytest

from sourced_memory import (
    AdmissionDecision,
    AuditEntry,
    UnknownChannelError,
    protect,
)
from sourced_memory.llm import MockLLM
from sourced_memory.router import LLMRouter, NullRouter


class FakeMem0:
    """Minimal Mem0-shaped stand-in for wrapped-store tests."""
    def __init__(self):
        self.added: list[dict] = []
    def add(self, message, *, user_id=None, metadata=None, **kw):
        self.added.append({"message": message, "user_id": user_id, "metadata": metadata, **kw})


# --- Basic behavior --------------------------------------------------------

def test_declared_channel_add_returns_audit_entry():
    memory = protect(trusted=["user"], untrusted=["web"])
    entry = memory.user.add("I love hiking.")
    assert isinstance(entry, AuditEntry)
    assert entry.decision is AdmissionDecision.BELIEF
    assert entry.source_name == "user"
    assert entry.trusted is True


def test_untrusted_personal_claim_rejected():
    """The core defense the library exists for."""
    mock = MockLLM({"default": {"functional_type": "personal_preference",
                                "confidence": 0.9, "supported": True}})
    memory = protect(
        trusted=["user"], untrusted=["web"],
        router=LLMRouter(mock),
    )
    memory.web.add("The user hates flying.")
    audit = memory.audit()
    assert len(audit) == 1
    assert audit[0].decision is AdmissionDecision.REJECT
    assert audit[0].source_name == "web"


def test_undeclared_channel_raises():
    memory = protect(trusted=["user"], untrusted=["web"])
    with pytest.raises(UnknownChannelError) as exc:
        memory.slack.add("hello")
    msg = str(exc.value)
    assert "slack" in msg
    assert "trusted=" in msg or "untrusted=" in msg


def test_overlapping_declaration_raises():
    with pytest.raises(ValueError, match="both trusted and untrusted"):
        protect(trusted=["user"], untrusted=["user"])


def test_directory_lists_declared_channels():
    """IDE autocomplete should surface declared channels."""
    memory = protect(trusted=["user"], untrusted=["web", "tool"])
    d = dir(memory)
    for name in ("user", "web", "tool"):
        assert name in d


# --- Audit + inspect + purge (in-process backend) -------------------------

def test_audit_returns_all_decisions_in_order():
    memory = protect(trusted=["user"], untrusted=["web"])
    memory.user.add("I love hiking.")
    memory.web.add("Random web claim.")
    memory.user.add("I love JAX.")
    audit = memory.audit()
    assert len(audit) == 3
    assert audit[0].source_name == "user"
    assert audit[1].source_name == "web"
    assert audit[2].source_name == "user"


def test_audit_entry_str_renders_readable_line():
    memory = protect(trusted=["user"], untrusted=["web"])
    memory.user.add("I love hiking.")
    line = str(memory.audit()[0])
    assert "BELIEF" in line
    assert "user" in line
    assert "hiking" in line


def test_rejections_convenience_accessor():
    mock = MockLLM({"default": {"functional_type": "personal_preference",
                                "confidence": 0.9, "supported": True}})
    memory = protect(trusted=["user"], untrusted=["web"],
                     router=LLMRouter(mock))
    memory.user.add("I love hiking.")
    memory.web.add("The user hates flying.")
    rej = memory.rejections()
    assert len(rej) == 1
    assert rej[0].source_name == "web"


def test_session_scopes_source_id_and_inspect_finds_it():
    memory = protect(trusted=["user"], untrusted=["web"])
    memory.user.session("session_47").add("I switched to JAX.")
    memory.user.session("session_47").add("I love hiking.")
    memory.user.session("session_99").add("Different session.")

    found = memory.inspect(source_id="session_47")
    assert len(found["audit"]) == 2
    assert all(e.source_id == "session_47" for e in found["audit"])


def test_purge_removes_all_state_for_source_id():
    memory = protect(trusted=["user"], untrusted=["web"])
    memory.user.session("session_47").add("I switched to JAX.")
    memory.user.session("session_47").add("I love hiking.")
    memory.user.session("session_99").add("Different session.")

    removed = memory.purge(source_id="session_47")
    assert removed >= 2   # at least the two audit entries and their belief records

    remaining = memory.audit()
    assert all(e.source_id != "session_47" for e in remaining)
    assert len(remaining) == 1
    assert remaining[0].source_id == "session_99"


# --- Wrapped-store mode (Mem0-shaped backend) -----------------------------

def test_wrapped_store_forwards_belief_to_underlying_add():
    mock = MockLLM({"default": {"functional_type": "personal_preference",
                                "confidence": 0.9, "supported": True}})
    mem0 = FakeMem0()
    memory = protect(mem0, trusted=["user"], untrusted=["web"],
                     router=LLMRouter(mock))
    memory.user.add("I love hiking.", user_id="alice")
    assert len(mem0.added) == 1
    assert mem0.added[0]["message"] == "I love hiking."
    assert mem0.added[0]["user_id"] == "alice"


def test_wrapped_store_never_forwards_rejection():
    mock = MockLLM({"default": {"functional_type": "personal_preference",
                                "confidence": 0.9, "supported": True}})
    mem0 = FakeMem0()
    memory = protect(mem0, trusted=["user"], untrusted=["web"],
                     router=LLMRouter(mock))
    memory.web.add("The user hates flying.", user_id="alice")
    assert mem0.added == []                          # never reached underlying store
    assert len(memory.rejections()) == 1              # but was recorded in the audit log


# --- Router / policy overrides -------------------------------------------

def test_default_router_is_rule_based():
    """protect() with no explicit router should still classify obvious phrasing."""
    memory = protect(trusted=["user"], untrusted=["web"])
    memory.user.add("I love hiking.")               # RuleBasedRouter catches 'i love'
    audit = memory.audit()
    assert audit[0].functional_type.value == "personal_preference"


def test_null_router_forces_a_type():
    from sourced_memory.models import FunctionalType
    memory = protect(
        trusted=["user"], untrusted=["web"],
        router=NullRouter(FunctionalType.EXTERNAL_FACT),
    )
    memory.user.add("Anything at all.")
    assert memory.audit()[0].functional_type is FunctionalType.EXTERNAL_FACT

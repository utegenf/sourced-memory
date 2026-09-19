"""Tests for the Decider facade + Mem0 adapter."""
from sourced_memory import (
    AdmissionDecision,
    Decider,
    FunctionalType,
    LLMRouter,
    NullRouter,
    TrustPolicy,
)
from sourced_memory.adapters.mem0 import wrap_mem0
from sourced_memory.llm import MockLLM


class FakeMem0:
    """Duck-typed Mem0 client for adapter tests."""
    def __init__(self):
        self.added: list[dict] = []
    def add(self, message, *, user_id=None, metadata=None, **kwargs):
        self.added.append({"message": message, "user_id": user_id, "metadata": metadata, **kwargs})
    def search(self, *args, **kwargs):  # pragma: no cover
        return []
    def get_all(self, *args, **kwargs):  # pragma: no cover
        return list(self.added)


# --- Decider ---------------------------------------------------------------

def test_decider_rejects_untrusted_personal_claim():
    llm = MockLLM({"default": {"functional_type": "personal_preference", "confidence": 0.9, "supported": True}})
    d = Decider(router=LLMRouter(llm), trusted_sources={"user"})
    r = d.decide("The user loves hiking.", source="external_document")
    assert r.decision is AdmissionDecision.REJECT
    assert r.source.trusted is False


def test_decider_accepts_trusted_personal_claim():
    llm = MockLLM({"default": {"functional_type": "personal_preference", "confidence": 0.9, "supported": True}})
    d = Decider(router=LLMRouter(llm), trusted_sources={"user"})
    r = d.decide("I love hiking.", source="user")
    assert r.decision is AdmissionDecision.BELIEF


def test_decider_routes_external_fact_to_candidate():
    llm = MockLLM({"default": {"functional_type": "external_fact", "confidence": 0.95, "supported": True}})
    d = Decider(router=LLMRouter(llm), trusted_sources={"user"})
    r = d.decide("Paris is the capital of France.", source="external_doc")
    assert r.decision is AdmissionDecision.CANDIDATE


def test_decider_never_stores_state():
    """The Decider is stateless; two identical decisions do not accumulate."""
    d = Decider(router=NullRouter(FunctionalType.PERSONAL_PREFERENCE), trusted_sources={"user"})
    d.decide("x", source="user")
    d.decide("x", source="user")
    assert not hasattr(d, "_beliefs")
    assert not hasattr(d, "_candidates")


# --- Mem0 adapter ----------------------------------------------------------

def test_wrap_mem0_forwards_belief_to_underlying_client():
    llm = MockLLM({"default": {"functional_type": "personal_preference", "confidence": 0.9, "supported": True}})
    mem0 = FakeMem0()
    wrapped = wrap_mem0(mem0, router=LLMRouter(llm), trusted_sources={"user"})
    r = wrapped.add("I love hiking.", user_id="alice", source="user")
    assert r.decision is AdmissionDecision.BELIEF
    assert len(mem0.added) == 1
    assert mem0.added[0]["message"] == "I love hiking."
    assert mem0.added[0]["user_id"] == "alice"
    # source enriched in metadata
    assert mem0.added[0]["metadata"]["source"] == "user"
    assert mem0.added[0]["metadata"]["functional_type"] == "personal_preference"


def test_wrap_mem0_does_not_forward_untrusted_personal_claim():
    """The attack: untrusted personal fabrication must NOT reach Mem0."""
    llm = MockLLM({"default": {"functional_type": "personal_preference", "confidence": 0.9, "supported": True}})
    mem0 = FakeMem0()
    wrapped = wrap_mem0(mem0, router=LLMRouter(llm), trusted_sources={"user"})
    r = wrapped.add("The user hates flying.", user_id="alice", source="external_document")
    assert r.decision is AdmissionDecision.REJECT
    assert mem0.added == []
    assert len(wrapped.rejections()) == 1


def test_wrap_mem0_holds_untrusted_world_fact_as_candidate():
    """A true external fact from an untrusted source is neither trusted nor discarded."""
    llm = MockLLM({"default": {"functional_type": "external_fact", "confidence": 0.95, "supported": True}})
    mem0 = FakeMem0()
    wrapped = wrap_mem0(mem0, router=LLMRouter(llm), trusted_sources={"user"})
    r = wrapped.add("Paris is the capital of France.", user_id="alice", source="external_doc")
    assert r.decision is AdmissionDecision.CANDIDATE
    assert mem0.added == []
    cs = wrapped.candidates()
    assert len(cs) == 1
    assert cs[0].source.name == "external_doc"


def test_wrap_mem0_holds_event_as_episodic():
    llm = MockLLM({"default": {"functional_type": "event", "confidence": 0.9, "supported": True}})
    mem0 = FakeMem0()
    wrapped = wrap_mem0(mem0, router=LLMRouter(llm), trusted_sources={"user"})
    r = wrapped.add("Please remind me to call Alice.", user_id="alice", source="user")
    assert r.decision is AdmissionDecision.EPISODIC
    # events never reach Mem0 by our reference policy
    assert mem0.added == []
    assert len(wrapped.episodic()) == 1

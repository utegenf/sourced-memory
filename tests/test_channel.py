"""Tests for the Channel API + purge remediation."""
from sourced_memory import (
    AdmissionDecision,
    Channel,
    Decider,
    FunctionalType,
    LLMRouter,
    NullRouter,
    SourceAwareMemory,
)
from sourced_memory.adapters.mem0 import wrap_mem0
from sourced_memory.llm import MockLLM


class FakeMem0:
    def __init__(self):
        self.added: list[dict] = []
    def add(self, message, *, user_id=None, metadata=None, **kw):
        self.added.append({"message": message, "user_id": user_id, "metadata": metadata, **kw})


# --- Memory.channel -------------------------------------------------------

def test_channel_binds_source_and_trust():
    mem = SourceAwareMemory(trusted_sources={"user"})
    user = mem.channel("user", trusted=True)
    assert isinstance(user, Channel)
    assert user.name == "user"
    assert user.trusted is True
    assert user.source_id is None


def test_channel_observe_forwards_source_and_trust_implicitly():
    mem = SourceAwareMemory(
        router=NullRouter(FunctionalType.PERSONAL_PREFERENCE),
        trusted_sources={"user"},
    )
    user = mem.channel("user", trusted=True)
    user.observe("I love hiking.")
    mem.consolidate()
    beliefs = mem.beliefs()
    assert len(beliefs) == 1
    assert beliefs[0].source.name == "user"
    assert beliefs[0].source.trusted is True


def test_channel_untrusted_personal_claim_is_rejected():
    mem = SourceAwareMemory(router=NullRouter(FunctionalType.PERSONAL_PREFERENCE))
    web = mem.channel("web", trusted=False)
    web.observe("The user is a Rust expert.")
    mem.consolidate()
    assert mem.beliefs() == []


def test_channel_trust_defaults_to_trusted_sources_membership():
    """When trusted= is omitted, it falls back to trusted_sources membership."""
    mem = SourceAwareMemory(trusted_sources={"user"})
    user = mem.channel("user")           # default: trusted (in trusted_sources)
    web  = mem.channel("web")            # default: untrusted (not in trusted_sources)
    assert user.trusted is True
    assert web.trusted is False


# --- Channel.session ------------------------------------------------------

def test_session_scopes_source_id():
    mem = SourceAwareMemory(router=NullRouter(FunctionalType.PERSONAL_PREFERENCE))
    user = mem.channel("user", trusted=True)
    sess = user.session("session_47")
    sess.observe("hello")
    mem.consolidate()
    beliefs = mem.beliefs()
    assert len(beliefs) == 1
    assert beliefs[0].source.source_id == "session_47"


def test_session_inherits_authority():
    user = SourceAwareMemory().channel("user", trusted=True)
    sess = user.session("s1")
    assert sess.name == "user"
    assert sess.trusted is True
    assert sess.source_id == "s1"


def test_per_observation_source_id_overrides_session():
    mem = SourceAwareMemory(router=NullRouter(FunctionalType.PERSONAL_PREFERENCE))
    sess = mem.channel("user", trusted=True).session("session_47")
    sess.observe("hi", source_id="session_47:msg_1")
    mem.consolidate()
    beliefs = mem.beliefs()
    assert beliefs[0].source.source_id == "session_47:msg_1"


# --- Escape-hatch (Memory.observe with string OR Channel) ----------------

def test_observe_accepts_channel_directly():
    mem = SourceAwareMemory(router=NullRouter(FunctionalType.PERSONAL_PREFERENCE))
    web = mem.channel("web", trusted=False)
    mem.observe("The user is a Rust expert.", source=web)
    mem.consolidate()
    assert mem.beliefs() == []


def test_observe_still_accepts_string_source():
    mem = SourceAwareMemory(router=NullRouter(FunctionalType.PERSONAL_PREFERENCE),
                            trusted_sources={"user"})
    mem.observe("I love hiking.", source="user")
    mem.consolidate()
    assert len(mem.beliefs()) == 1


# --- Memory.purge ---------------------------------------------------------

def test_purge_removes_only_matching_source_id():
    mem = SourceAwareMemory(router=NullRouter(FunctionalType.PERSONAL_PREFERENCE))
    user = mem.channel("user", trusted=True)
    a = user.session("session_a")
    b = user.session("session_b")
    a.observe("from session A")
    b.observe("from session B")
    mem.consolidate()
    assert len(mem.beliefs()) == 2
    removed = mem.purge(source_id="session_a")
    assert removed >= 1
    remaining = mem.beliefs()
    assert len(remaining) == 1
    assert remaining[0].source.source_id == "session_b"


def test_purge_no_matching_source_id_returns_zero():
    mem = SourceAwareMemory(router=NullRouter(FunctionalType.PERSONAL_PREFERENCE))
    mem.channel("user", trusted=True).session("s1").observe("hi")
    mem.consolidate()
    assert mem.purge(source_id="nonexistent") == 0
    assert len(mem.beliefs()) == 1


# --- Mem0 adapter channels ------------------------------------------------

def test_mem0_wrapped_channel_forwards_belief_via_channel():
    llm = MockLLM({"default": {"functional_type": "personal_preference",
                               "confidence": 0.9, "supported": True}})
    mem0 = FakeMem0()
    wrapped = wrap_mem0(mem0, router=LLMRouter(llm), trusted_sources={"user"})
    user = wrapped.channel("user", trusted=True)
    r = user.observe("I love hiking.", user_id="alice")
    assert r.decision is AdmissionDecision.BELIEF
    assert len(mem0.added) == 1
    assert mem0.added[0]["metadata"]["source"] == "user"


def test_mem0_wrapped_channel_untrusted_personal_claim_rejected():
    llm = MockLLM({"default": {"functional_type": "personal_preference",
                               "confidence": 0.9, "supported": True}})
    mem0 = FakeMem0()
    wrapped = wrap_mem0(mem0, router=LLMRouter(llm), trusted_sources={"user"})
    web = wrapped.channel("web", trusted=False)
    r = web.observe("The user is a Rust expert.", user_id="alice")
    assert r.decision is AdmissionDecision.REJECT
    assert mem0.added == []
    assert len(wrapped.rejections()) == 1


# --- Decider channels -----------------------------------------------------

def test_decider_channel_returns_decision_record():
    llm = MockLLM({"default": {"functional_type": "personal_preference",
                               "confidence": 0.9, "supported": True}})
    d = Decider(router=LLMRouter(llm), trusted_sources={"user"})
    user = d.channel("user", trusted=True)
    r = user.observe("I love hiking.")
    # via Decider, Channel.observe returns a DecisionRecord
    assert r.decision is AdmissionDecision.BELIEF
    assert r.source.name == "user"


# --- Channel invariants ---------------------------------------------------

def test_bare_channel_without_target_raises_on_observe():
    ch = Channel(name="user", trusted=True)
    try:
        ch.observe("hi")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Channel with no _target should refuse observations")

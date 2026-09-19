"""Tests for the LLM-backed router and the MockLLM stub."""
from sourced_memory import LLMRouter, NullRouter, SourceAwareMemory
from sourced_memory.llm import LLMResponseError, MockLLM
from sourced_memory.models import FunctionalType


def test_llm_router_returns_declared_functional_type():
    llm = MockLLM({"default": {
        "functional_type": "personal_preference",
        "confidence": 0.9,
        "supported": True,
        "summarized_content": "user prefers hiking",
    }})
    r = LLMRouter(llm)
    result = r.route("I love hiking on weekends.")
    assert result.functional_type is FunctionalType.PERSONAL_PREFERENCE
    assert result.confidence == 0.9
    assert result.supported is True
    assert result.summarized_content == "user prefers hiking"


def test_llm_router_never_sees_source_metadata():
    """Invariant: the LLM is only handed content. If the router leaked source
    metadata into the LLM call we'd see it in ``captured['user']``."""
    captured = {}

    def classify(system, user, schema, name):
        captured["system"] = system
        captured["user"] = user
        return {
            "functional_type": "personal_preference",
            "confidence": 0.9,
            "supported": True,
        }

    mem = SourceAwareMemory(
        trusted_sources={"user"},
        router=LLMRouter(MockLLM(classify)),
    )
    mem.observe("I love hiking.", source="secret_untrusted_channel_alice_123")
    mem.consolidate()
    assert "secret_untrusted_channel_alice_123" not in captured.get("user", "")
    assert "secret_untrusted_channel_alice_123" not in captured.get("system", "")


def test_llm_router_untrusted_personal_claim_is_rejected():
    """End-to-end: LLM says 'personal_preference'; untrusted source ⇒ REJECT."""
    llm = MockLLM({"default": {
        "functional_type": "personal_preference",
        "confidence": 0.95,
        "supported": True,
    }})
    mem = SourceAwareMemory(trusted_sources={"user"}, router=LLMRouter(llm))
    mem.observe("The user hates flying.", source="external_document")
    mem.consolidate()
    assert mem.beliefs() == []


def test_llm_router_raises_on_bad_response():
    llm = MockLLM({})  # empty handler
    r = LLMRouter(llm)
    try:
        r.route("some content")
    except LLMResponseError:
        pass
    else:
        raise AssertionError("expected LLMResponseError for missing handler")


def test_null_router_emits_fixed_type():
    """NullRouter emits the configured type verbatim without any classification."""
    router = NullRouter(FunctionalType.EXTERNAL_FACT, confidence=1.0)
    result = router.route("anything at all")
    assert result.functional_type is FunctionalType.EXTERNAL_FACT
    assert result.confidence == 1.0
    assert result.summarized_content == "anything at all"


def test_null_router_default_is_external_fact():
    result = NullRouter().route("something")
    assert result.functional_type is FunctionalType.EXTERNAL_FACT


def test_mock_llm_records_calls():
    calls_seen = []

    def h(system, user, schema, name):
        calls_seen.append((system[:20], user[:20]))
        return {"functional_type": "event", "confidence": 1.0, "supported": True}

    llm = MockLLM(h)
    LLMRouter(llm).route("Please remind me to call Alice.")
    assert len(calls_seen) == 1
    assert len(llm.calls) == 1
    assert "Please remind me" in llm.calls[0]["user"]

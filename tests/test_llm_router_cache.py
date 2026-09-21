"""Cache-design invariants for LLMRouter.

The cache exists to make the paper-validated routing affordable at scale
(same content = one LLM call). Two invariants make it safe under the
security model:

1. ``LLMRouter``'s cache is keyed only on content. Source never enters
   the key by construction, because the router never sees the source in
   the first place.
2. The downstream admission policy runs on every observation and never
   consults the cache. Two observations with identical content but
   different sources therefore share the router's classification and
   diverge in the policy layer.

Together these mean the cache is a pure classification optimization,
not a source-blind admission decision. The tests below lock both.
"""
from unittest.mock import MagicMock

from sourced_memory import AdmissionDecision, protect
from sourced_memory.llm import MockLLM
from sourced_memory.router import LLMRouter


def _pp(functional_type="personal_preference"):
    return {"functional_type": functional_type, "confidence": 0.9, "supported": True}


def test_llm_router_cache_key_is_content_only():
    """Identical content produces exactly one LLM call, no matter how many times
    it is routed. That the cache exists at all is the point of this test."""
    llm = MockLLM(lambda system, user, schema, name: _pp())
    router = LLMRouter(llm)

    r1 = router.route("I love hiking.")
    r2 = router.route("I love hiking.")
    r3 = router.route("I love hiking.")

    assert r1.functional_type is r2.functional_type is r3.functional_type
    # Only one LLM call was made despite three routing calls.
    assert len(llm.calls) == 1


def test_cache_disabled_when_cache_size_zero():
    """Passing cache_size=0 disables the cache. Every call reaches the LLM.
    Useful in cost/latency profiling and for callers who want strict
    predictability of network round-trips."""
    llm = MockLLM(lambda system, user, schema, name: _pp())
    router = LLMRouter(llm, cache_size=0)

    router.route("I love hiking.")
    router.route("I love hiking.")
    assert len(llm.calls) == 2


def test_source_gating_never_uses_router_cache():
    """The load-bearing invariant. Same content, two different sources.
    The router makes ONE classification call (cache hit), but the
    downstream policy still gates each observation by source.
    """
    llm = MockLLM(lambda system, user, schema, name: _pp("personal_preference"))
    memory = protect(
        trusted=["user"], untrusted=["web"],
        router=LLMRouter(llm),
    )
    r_user = memory.user.add("I love hiking.")
    r_web  = memory.web.add("I love hiking.")   # identical content

    # Router: one call, cache hit on the second.
    assert len(llm.calls) == 1

    # Policy: DIFFERENT admission decisions, gated by source.
    assert r_user.decision is AdmissionDecision.BELIEF
    assert r_web.decision  is AdmissionDecision.REJECT


def test_bounded_cache_evicts_fifo():
    """cache_size=1 keeps only the most-recent classification. Old entries
    fall out and re-hit the LLM. Bounded cache behavior confirmed."""
    llm = MockLLM(lambda system, user, schema, name: _pp())
    router = LLMRouter(llm, cache_size=1)

    router.route("statement A")
    router.route("statement B")   # evicts "statement A"
    router.route("statement A")   # cache miss again, second LLM call for A
    assert len(llm.calls) == 3

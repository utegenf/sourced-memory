"""Tests for the loud-degradation posture on router selection.

The security-relevant invariant is that silence is never a valid state:
- No router argument -> RoutingDegradedWarning (weak-by-default is loud).
- Explicit RuleBasedRouter() -> one-time INFO log (informed choice, named).
- Explicit LLMRouter or another paper-quality router -> completely silent.
"""
import logging
import warnings

import pytest

from sourced_memory import RoutingDegradedWarning, protect
from sourced_memory.router import LLMRouter, RuleBasedRouter


def test_default_router_emits_RoutingDegradedWarning():
    """`protect()` without router= is the silent-default trap; warn loudly."""
    with pytest.warns(RoutingDegradedWarning) as records:
        protect(trusted=["user"], untrusted=["web"])
    assert len(records) == 1
    msg = str(records[0].message)
    assert "RuleBasedRouter" in msg
    assert "third-person" in msg
    assert "LLMRouter" in msg   # actionable path back to paper-quality


def test_explicit_RuleBasedRouter_does_NOT_emit_warning(caplog):
    """User made an informed choice; no warning, one INFO log per process."""
    # Reset the module-level notified flag so this test is self-contained.
    # ``sourced_memory.protect`` as an attribute of the package resolves to
    # the ``protect()`` function (which __init__.py re-exports and shadows
    # the module name). Grab the module itself via importlib.
    import importlib
    p = importlib.import_module("sourced_memory.protect")
    p._explicit_rulerouter_notified = False

    with warnings.catch_warnings():
        warnings.simplefilter("error", RoutingDegradedWarning)   # would raise if fired
        with caplog.at_level(logging.INFO, logger="sourced_memory"):
            protect(trusted=["user"], untrusted=["web"],
                    router=RuleBasedRouter())
    assert any("RuleBasedRouter explicitly" in r.message for r in caplog.records)


def test_explicit_RuleBasedRouter_INFO_fires_only_once_per_process(caplog):
    # ``sourced_memory.protect`` as an attribute of the package resolves to
    # the ``protect()`` function (which __init__.py re-exports and shadows
    # the module name). Grab the module itself via importlib.
    import importlib
    p = importlib.import_module("sourced_memory.protect")
    p._explicit_rulerouter_notified = False

    caplog.set_level(logging.INFO, logger="sourced_memory")
    protect(trusted=["user"], router=RuleBasedRouter())
    protect(trusted=["user"], router=RuleBasedRouter())
    protect(trusted=["user"], router=RuleBasedRouter())
    matching = [r for r in caplog.records if "RuleBasedRouter explicitly" in r.message]
    assert len(matching) == 1


def test_paper_quality_router_is_silent(caplog):
    """Explicit LLMRouter (or any non-RuleBasedRouter) is fully silent."""
    from sourced_memory.llm import MockLLM
    fake_llm = MockLLM({"default": {"functional_type": "personal_preference",
                                    "confidence": 0.9, "supported": True}})
    with warnings.catch_warnings():
        warnings.simplefilter("error", RoutingDegradedWarning)
        with caplog.at_level(logging.INFO, logger="sourced_memory"):
            protect(trusted=["user"], router=LLMRouter(fake_llm))
    # No routing-degraded-related INFO log either.
    assert not any("RuleBasedRouter" in r.message for r in caplog.records)

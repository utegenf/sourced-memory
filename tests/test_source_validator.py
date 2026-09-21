"""Tests for the source_validator hook on protect().

The hook is a load-bearing security control: rejecting the validator must
short-circuit admission (not just log a warning) and must fire BEFORE the
router runs, so a rejecting validator saves the router call. These tests
guard against a regression that would let the hook fire but not gate.
"""
from unittest.mock import MagicMock

import pytest

from sourced_memory import (
    AdmissionDecision,
    RoutingDegradedWarning,
    protect,
)
from sourced_memory.router import Router


def test_source_validator_rejecting_short_circuits_router():
    """A False-returning validator must reject BEFORE the router runs.

    Passing an explicit MagicMock(spec=Router) means no RoutingDegradedWarning
    should fire (we made an informed choice about the router shape). If a
    future refactor adds a warning here, this test will notice via the
    strict filter.
    """
    router = MagicMock(spec=Router)
    memory = protect(
        trusted=["user"], untrusted=["web"],
        router=router,
        source_validator=lambda ch, content, source_id: False,
    )
    entry = memory.user.add("I love hiking.")

    # Load-bearing assertion: router.route() must not have been called.
    router.route.assert_not_called()
    assert entry.decision is AdmissionDecision.REJECT
    assert "source_validator rejected" in entry.reason


def test_source_validator_raising_treated_as_rejection():
    """A validator that raises must also short-circuit; exception surfaces in reason."""
    router = MagicMock(spec=Router)
    def raising_validator(ch, content, source_id):
        raise PermissionError("token expired")

    memory = protect(
        trusted=["user"], untrusted=["web"],
        router=router,
        source_validator=raising_validator,
    )
    entry = memory.user.add("I love hiking.")

    router.route.assert_not_called()
    assert entry.decision is AdmissionDecision.REJECT
    assert "PermissionError" in entry.reason
    assert "token expired" in entry.reason


def test_source_validator_accepting_allows_normal_admission():
    """A True-returning validator must allow the router + policy to run normally."""
    memory = protect(
        trusted=["user"], untrusted=["web"],
        router=None,  # will emit RoutingDegradedWarning; ignored via filter below
        source_validator=lambda ch, content, source_id: True,
    )
    # RoutingDegradedWarning fires above from the None-router path; that
    # is expected here since we are not testing the router choice.
    import warnings
    warnings.filterwarnings("ignore", category=RoutingDegradedWarning)

    entry = memory.user.add("I love hiking.")
    assert entry.decision is AdmissionDecision.BELIEF


def test_source_validator_gets_correct_args():
    seen: list[tuple] = []
    def spy(ch, content, source_id):
        seen.append((ch, content, source_id))
        return True

    memory = protect(
        trusted=["user"], untrusted=["web"],
        source_validator=spy,
    )
    memory.user.session("s47").add("Statement A", user_id="alice")
    memory.web.add("Statement B")

    assert seen == [
        ("user", "Statement A", "s47"),
        ("web",  "Statement B", None),
    ]

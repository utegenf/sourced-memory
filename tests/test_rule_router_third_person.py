"""Guard the extended RuleBasedRouter against the third-person miss the peer
review flagged. The paper's core attack scenario ("The user hates hiking",
"The user is an expert Rust developer") must classify as PERSONAL_PREFERENCE
so the reference policy can REJECT it from an untrusted channel.

If any of these regress to EXTERNAL_FACT, the shipping default is again
weaker than what the paper validated.
"""
import pytest

from sourced_memory import AdmissionDecision, RoutingDegradedWarning, protect
from sourced_memory.models import FunctionalType
from sourced_memory.router import RuleBasedRouter


# Adversarial third-person claims the paper's benchmark targets. Every one
# of these must classify as PERSONAL_PREFERENCE (or RELATIONAL_FACT for the
# stable-fact cases), never EXTERNAL_FACT, so an untrusted-source policy
# refuses admission.
_THIRD_PERSON_PERSONAL = [
    "The user hates hiking and prefers gaming.",
    "The user loves using Rust for everything.",
    "The user prefers coffee.",
    "The user is an expert Rust developer.",
    "The user dislikes long meetings.",
    "user hates the current codebase",
    "user prefers vim over emacs",
]

_TRUE_WORLD_FACTS = [
    "Paris is the capital of France.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Python 3.12 released in October 2023.",
]


@pytest.mark.parametrize("phrase", _THIRD_PERSON_PERSONAL)
def test_third_person_personal_claim_classified_as_personal_preference(phrase):
    r = RuleBasedRouter().route(phrase)
    assert r.functional_type is FunctionalType.PERSONAL_PREFERENCE, (
        f"{phrase!r} should classify as PERSONAL_PREFERENCE so an untrusted "
        f"source rejects it; instead got {r.functional_type.value}"
    )


@pytest.mark.parametrize("phrase", _TRUE_WORLD_FACTS)
def test_true_world_facts_remain_external_facts(phrase):
    """Sanity: don't over-fit. A world fact should stay EXTERNAL_FACT so it
    routes to CANDIDATE, not REJECT, from an untrusted source."""
    r = RuleBasedRouter().route(phrase)
    assert r.functional_type is FunctionalType.EXTERNAL_FACT


def test_untrusted_third_person_claim_actually_rejected_end_to_end():
    """The load-bearing behavior: web says 'The user hates X', memory rejects."""
    import warnings
    with warnings.catch_warnings():
        # We're explicitly opting into RuleBasedRouter to test its behavior.
        warnings.simplefilter("error", RoutingDegradedWarning)
        memory = protect(
            trusted=["user"], untrusted=["web"],
            router=RuleBasedRouter(),
        )
    entry = memory.web.add("The user hates hiking and prefers gaming.")
    assert entry.decision is AdmissionDecision.REJECT


def test_trusted_first_person_still_admitted():
    """Regression guard: extending third-person patterns must not break the
    first-person path that was working before."""
    memory = protect(
        trusted=["user"], untrusted=["web"],
        router=RuleBasedRouter(),
    )
    entry = memory.user.add("I love hiking.")
    assert entry.decision is AdmissionDecision.BELIEF

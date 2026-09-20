from sourced_memory import AdmissionDecision, FunctionalType, TrustPolicy
from sourced_memory.models import Source

def test_reference_policy_trusts_personal_claims_from_trusted_source():
    assert TrustPolicy.reference().decide(Source("user", trusted=True), FunctionalType.PERSONAL_PREFERENCE) is AdmissionDecision.BELIEF

def test_reference_policy_rejects_personal_claims_from_untrusted_source():
    assert TrustPolicy.reference().decide(Source("document"), FunctionalType.PERSONAL_PREFERENCE) is AdmissionDecision.REJECT

def test_reference_policy_keeps_external_facts_as_candidates():
    assert TrustPolicy.reference().decide(Source("document"), FunctionalType.EXTERNAL_FACT) is AdmissionDecision.CANDIDATE

def test_source_specific_rule_overrides_trust_default():
    policy = TrustPolicy({("document", FunctionalType.PERSONAL_PREFERENCE): AdmissionDecision.BELIEF})
    assert policy.decide(Source("document"), FunctionalType.PERSONAL_PREFERENCE) is AdmissionDecision.BELIEF

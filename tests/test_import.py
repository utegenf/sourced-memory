def test_public_imports():
    from sourced_memory import SourceAwareMemory, SourceTypePolicy, FunctionalType, AdmissionDecision
    assert SourceAwareMemory
    assert SourceTypePolicy
    assert FunctionalType
    assert AdmissionDecision

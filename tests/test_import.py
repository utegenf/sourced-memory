def test_public_imports():
    """Everything the README documents should be importable from the top level."""
    from sourced_memory import (
        protect,
        ProtectedMemory,
        AuditEntry,
        UnknownChannelError,
        AdmissionDecision,
        FunctionalType,
        Source,
        Belief,
        CandidateEvidence,
        Experience,
        TrustPolicy,
        TrustPolicy,
        Router,
        RouteResult,
        LLMRouter,
        RuleBasedRouter,
        NullRouter,
        CallableRouter,
    )
    assert protect
    assert ProtectedMemory
    assert AuditEntry
    assert UnknownChannelError


def test_advanced_imports():
    """Low-level primitives are reachable via sourced_memory.advanced."""
    from sourced_memory.advanced import (
        SourceAwareMemory,
        Channel,
        Decider,
        DecisionRecord,
        AdmissionRecord,
        WrappedMem0,
        wrap_mem0,
    )
    assert SourceAwareMemory
    assert Channel
    assert Decider


def test_version():
    import sourced_memory
    assert sourced_memory.__version__ == "0.1.0a6"

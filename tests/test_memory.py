from sourced_memory.advanced import SourceAwareMemory

def test_untrusted_personal_claim_does_not_become_belief():
    memory = SourceAwareMemory(trusted_sources={"user"})
    memory.observe("I love hiking.", source="document")
    memory.consolidate()
    assert memory.beliefs() == []

def test_trusted_personal_claim_becomes_belief():
    memory = SourceAwareMemory(trusted_sources={"user"})
    memory.observe("I love hiking.", source="user")
    memory.consolidate()
    assert len(memory.beliefs()) == 1
    assert memory.beliefs()[0].source.name == "user"

def test_untrusted_external_fact_becomes_candidate():
    memory = SourceAwareMemory(trusted_sources={"user"})
    memory.observe("The capital of France is Paris.", source="document")
    memory.consolidate()
    assert len(memory.candidates()) == 1
    assert memory.candidates()[0].source.name == "document"

def test_provenance_is_not_inferred_from_content():
    memory = SourceAwareMemory(trusted_sources={"user"})
    memory.observe("The user loves hiking.", source="document")
    memory.consolidate()
    assert memory.beliefs() == []

def test_event_is_episodic():
    memory = SourceAwareMemory(trusted_sources={"user"})
    memory.observe("Please remind me to call Alice.", source="user")
    memory.consolidate()
    assert len(memory.episodic()) == 1

def test_readers_do_not_trigger_consolidation():
    memory = SourceAwareMemory(trusted_sources={"user"})
    memory.observe("I love hiking.", source="user")
    # No explicit consolidate() call: readers must not advance state.
    assert memory.beliefs() == []
    assert memory.pending() == 1
    memory.consolidate()
    assert len(memory.beliefs()) == 1
    assert memory.pending() == 0

from sourced_memory import AdmissionDecision, FunctionalType, SourceAwareMemory, SourceTypePolicy

policy = SourceTypePolicy({
    ("user", FunctionalType.PERSONAL_PREFERENCE): AdmissionDecision.BELIEF,
    ("user", FunctionalType.GENERAL_RULE): AdmissionDecision.BELIEF,
    ("user", FunctionalType.RELATIONAL_FACT): AdmissionDecision.BELIEF,
    ("document", FunctionalType.PERSONAL_PREFERENCE): AdmissionDecision.REJECT,
    ("document", FunctionalType.EXTERNAL_FACT): AdmissionDecision.CANDIDATE,
    ("tool", FunctionalType.EXTERNAL_FACT): AdmissionDecision.CANDIDATE,
    ("user", FunctionalType.EVENT): AdmissionDecision.EPISODIC,
})

memory = SourceAwareMemory(policy=policy)
memory.observe("I prefer tea.", source="user")
memory.observe("The user prefers tea.", source="document")
memory.observe("Paris is the capital of France.", source="tool")
memory.consolidate()

print("Beliefs:", [x.content for x in memory.beliefs()])
print("Candidates:", [x.content for x in memory.candidates()])

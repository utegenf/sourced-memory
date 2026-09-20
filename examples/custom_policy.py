"""Custom trust policy via protect().

The default TrustPolicy.reference() implements the paper's rules. Any
application can substitute its own (source × functional-type) rule table.
Rules are checked most-specific to least-specific: explicit ``(name, type)``
first, then ``("*", type)`` wildcards, then ``("trusted"/"untrusted", type)``
by trust level, then the default (``REJECT``).
"""
from sourced_memory import AdmissionDecision, FunctionalType, TrustPolicy, protect


custom = TrustPolicy({
    # Personal claims from the user become beliefs; from documents, rejected.
    ("user",     FunctionalType.PERSONAL_PREFERENCE): AdmissionDecision.BELIEF,
    ("user",     FunctionalType.GENERAL_RULE):        AdmissionDecision.BELIEF,
    ("user",     FunctionalType.RELATIONAL_FACT):     AdmissionDecision.BELIEF,
    ("document", FunctionalType.PERSONAL_PREFERENCE): AdmissionDecision.REJECT,
    # External facts from any untrusted source go to the candidate layer.
    ("document", FunctionalType.EXTERNAL_FACT):       AdmissionDecision.CANDIDATE,
    ("tool",     FunctionalType.EXTERNAL_FACT):       AdmissionDecision.CANDIDATE,
    # Events are transient by default.
    ("user",     FunctionalType.EVENT):               AdmissionDecision.EPISODIC,
})

memory = protect(
    trusted   = ["user"],
    untrusted = ["document", "tool"],
    policy    = custom,
)

memory.user.add("I prefer tea.")
memory.document.add("The user prefers tea.")
memory.tool.add("Paris is the capital of France.")

for entry in memory.audit():
    print(entry)

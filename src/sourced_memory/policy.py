"""Configurable source x functional-type admission policy."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Union
from .models import AdmissionDecision, FunctionalType, Source

DecisionLike = Union[AdmissionDecision, str]

class SourceTypePolicy:
    """Deterministic, auditable policy for the source/type trust boundary."""

    def __init__(self, rules: Mapping[tuple[str, str | FunctionalType], DecisionLike] | None = None,
                 *, default: DecisionLike = AdmissionDecision.REJECT):
        self._rules = {
            (source_name, self._type_key(functional_type)): self._decision(decision)
            for (source_name, functional_type), decision in (rules or {}).items()
        }
        self.default = self._decision(default)

    @staticmethod
    def _type_key(value: str | FunctionalType) -> str:
        return value.value if isinstance(value, FunctionalType) else value

    @staticmethod
    def _decision(value: DecisionLike) -> AdmissionDecision:
        return value if isinstance(value, AdmissionDecision) else AdmissionDecision(value)

    def decide(self, source: Source, functional_type: FunctionalType) -> AdmissionDecision:
        key = functional_type.value
        for candidate in ((source.name, key), ("*", key),
                          ("trusted" if source.trusted else "untrusted", key)):
            if candidate in self._rules:
                return self._rules[candidate]
        return self.default

    @classmethod
    def reference(cls) -> "SourceTypePolicy":
        personal = (
            FunctionalType.PERSONAL_PREFERENCE.value,
            FunctionalType.GENERAL_RULE.value,
            FunctionalType.RELATIONAL_FACT.value,
        )
        rules = {}
        for type_name in personal:
            rules[("trusted", type_name)] = AdmissionDecision.BELIEF
            rules[("untrusted", type_name)] = AdmissionDecision.REJECT
        rules[("*", FunctionalType.EXTERNAL_FACT.value)] = AdmissionDecision.CANDIDATE
        rules[("*", FunctionalType.EVENT.value)] = AdmissionDecision.EPISODIC
        return cls(rules)

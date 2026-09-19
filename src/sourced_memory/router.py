"""Content-only functional routing.

Routers see the item text and nothing else. Provenance is applied afterward by
the policy; a router implementation cannot access `Source` even by mistake.
"""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from .models import FunctionalType

@dataclass(frozen=True)
class RouteResult:
    functional_type: FunctionalType
    confidence: float = 1.0
    supported: bool = True
    summarized_content: str | None = None

class Router(Protocol):
    def route(self, content: str) -> RouteResult: ...

class CallableRouter:
    def __init__(self, classifier: Callable[[str], RouteResult]):
        self._classifier = classifier
    def route(self, content: str) -> RouteResult:
        return self._classifier(content)

class RuleBasedRouter:
    """Dependency-free example router; use an LLM-backed Router for production."""
    def route(self, content: str) -> RouteResult:
        text = content.lower().strip()
        if text.endswith("?") or text.startswith(("please ", "can you ", "could you ", "remind me")):
            return RouteResult(FunctionalType.EVENT, summarized_content=content)
        if any(x in text for x in ("i like ", "i love ", "i prefer ", "i dislike ", "i hate ", "my favorite ")):
            return RouteResult(FunctionalType.PERSONAL_PREFERENCE, summarized_content=content)
        if any(x in text for x in ("my ", "our ", "i live ", "i work ", "my partner", "my manager")):
            return RouteResult(FunctionalType.RELATIONAL_FACT, summarized_content=content)
        if any(x in text for x in ("always ", "never ", "from now on", "please always", "please never")):
            return RouteResult(FunctionalType.GENERAL_RULE, summarized_content=content)
        return RouteResult(FunctionalType.EXTERNAL_FACT, summarized_content=content)

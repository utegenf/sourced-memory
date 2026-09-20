import inspect
import typing

from sourced_memory.advanced import SourceAwareMemory
from sourced_memory.models import FunctionalType
from sourced_memory.router import CallableRouter, Router, RouteResult

def test_router_protocol_takes_content_str_only():
    """Structural invariant: Router.route accepts a string, not an Experience."""
    hints = typing.get_type_hints(Router.route)
    sig = inspect.signature(Router.route)
    params = [name for name in sig.parameters if name != "self"]
    assert params == ["content"]
    assert hints["content"] is str

def test_router_receives_content_not_source():
    seen = {}
    def classify(content):
        seen["content"] = content
        return RouteResult(FunctionalType.PERSONAL_PREFERENCE, summarized_content=content)
    memory = SourceAwareMemory(trusted_sources={"user"}, router=CallableRouter(classify))
    memory.observe("I prefer tea.", source="document")
    memory.consolidate()
    assert seen == {"content": "I prefer tea."}
    assert memory.beliefs() == []

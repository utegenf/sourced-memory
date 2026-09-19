"""Helpers for explicit application-supplied provenance."""
from __future__ import annotations
from .models import Source

def source(name: str, *, source_id: str | None = None, trusted: bool = False, **metadata) -> Source:
    return Source(name=name, source_id=source_id, trusted=trusted, metadata=metadata)

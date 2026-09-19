"""Adapters that put sourced-memory in front of existing memory systems.

Each adapter is a reference integration, not a hard dependency of the core
package. Users install adapters via optional extras, e.g.::

    pip install "sourced-memory[mem0]"
"""
from __future__ import annotations

__all__ = ["mem0"]

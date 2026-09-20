"""Low-level building blocks used by ``protect()`` under the hood.

These are exposed for advanced use cases:

- Writing your own adapter for a memory system that ``protect()`` does not
  auto-detect yet.
- Framework integrations that need direct access to the admission decision
  function (``Decider``).
- Runtime-dynamic channel names, where a set of trusted/untrusted sources is
  not known at ``protect()`` construction time.
- Reproducing the paper's benchmark, which uses the raw
  :class:`SourceAwareMemory` two-phase model.

For everyday application use, prefer :func:`sourced_memory.protect`.
"""
from __future__ import annotations

from .adapters.mem0 import WrappedMem0, wrap_mem0
from .channel import Channel
from .decider import Decider, DecisionRecord
from .memory import AdmissionRecord, SourceAwareMemory

__all__ = [
    "AdmissionRecord",
    "Channel",
    "Decider",
    "DecisionRecord",
    "SourceAwareMemory",
    "WrappedMem0",
    "wrap_mem0",
]

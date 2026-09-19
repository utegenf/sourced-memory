"""Channels: preconfigured source bindings for admission.

A ``Channel`` binds an origin (source name + trust flag + optional source_id)
to a target that accepts observations. Applications configure channels
*once* at setup and reuse them; callers then observe through the channel
without repeating the source metadata on every call.

Motivation
----------
The bare ``memory.observe(text, source="user")`` API is easy to misuse:
every call site has to remember (a) that sources exist, (b) which string is
correct, and (c) to pass it. Channels move that decision to configuration
time and make omission a type error instead of a silent trust downgrade.

Design
------
Channel is an ergonomic wrapper. It holds ``name``, ``trusted``, and an
optional default ``source_id``, and forwards to a target object that
implements one of two shapes:

    target.observe(content, *, source, source_id, trusted, metadata)
    target.add(content,      *, source, source_id, trusted, metadata, **kw)

Both are supported: ``SourceAwareMemory`` exposes the first, the
``WrappedMem0`` adapter exposes the second. Channel dispatches to whichever
is available, so the same abstraction works for standalone memory and for
adapters sitting in front of an existing store.

Source identity
---------------
``channel.session(source_id)`` returns a *scoped* Channel that carries a
specific ``source_id``. Same authority, distinct origin identity. This is
what enables targeted remediation later (``memory.purge(source_id=...)``).
Per-observation overrides are allowed:

    user   = memory.channel("user", trusted=True)
    sess47 = user.session("session_47")
    sess47.observe("hello")                                  # source_id=session_47
    sess47.observe("hello", source_id="session_47:msg_1")    # explicit override

The escape hatch ``memory.observe(text, source=channel_or_str)`` remains
available for dynamic cases where the channel is chosen at runtime.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Channel:
    """Preconfigured source binding for admission.

    Immutable. Applications typically create a small number of these at
    setup (``user``, ``web``, ``internal_tool``) and reuse them everywhere
    an observation is made.
    """
    name: str
    trusted: bool
    source_id: str | None = None
    _target: Any = None      # SourceAwareMemory, WrappedMem0, or duck-typed

    def session(self, source_id: str) -> "Channel":
        """Return a Channel with the same authority but a scoped source_id.

        Useful for per-user-session or per-request identity that later
        enables targeted remediation via ``memory.purge(source_id=...)``.
        """
        return Channel(
            name=self.name,
            trusted=self.trusted,
            source_id=source_id,
            _target=self._target,
        )

    def observe(
        self,
        content: str,
        *,
        source_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Observe through this channel.

        The target is called with the channel's name, trust, and source_id
        bound implicitly. ``source_id`` overrides the channel's default;
        ``**kwargs`` (e.g. ``user_id`` for Mem0) passes through unchanged.
        """
        if self._target is None:
            raise RuntimeError(
                "Channel has no target; obtain channels from memory.channel(...) "
                "or wrapped_adapter.channel(...) rather than instantiating Channel directly."
            )
        sid = source_id if source_id is not None else self.source_id
        # Prefer .observe (Memory / Decider); fall back to .add (adapters).
        for method_name in ("observe", "add"):
            method = getattr(self._target, method_name, None)
            if callable(method):
                return method(
                    content,
                    source=self.name,
                    source_id=sid,
                    trusted=self.trusted,
                    metadata=metadata,
                    **kwargs,
                )
        raise TypeError(
            f"Channel target {type(self._target).__name__} has neither "
            f".observe(...) nor .add(...) methods"
        )

    # Convenience alias so a Channel can stand in wherever adapters expect .add
    def add(self, content: str, **kwargs: Any) -> Any:
        return self.observe(content, **kwargs)

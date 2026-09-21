"""Tests for the WrappedMem0 loop-delete purge.

The remediation story is only complete when the underlying backing store
actually loses the record on purge. Mem0's open-source Memory class exposes
only ``delete(memory_id)`` (no metadata filter), so purge must loop over
the ids the adapter captured on write. The tests below lock:

- The right ids are passed to ``store.delete`` (from the audit log's
  ``backing_store_id`` field).
- Partial failure does not silently mark records as purged; the failed
  entries stay in the audit log and are visible in ``PurgeResult``.
- A subsequent purge retries the failed entries. No permanent orphan.
- If the store's ``.add()`` returned no id we can recognize, the record
  is counted as ``records_unreachable_in_backing_store`` and this is
  named as a gap, not silently treated as success.

The mem0 duck-typed client used here returns responses shaped like the
real open-source ``Memory.add``:
    {"results": [{"id": "<uuid>", "memory": "...", "event": "ADD"}]}
"""
from unittest.mock import MagicMock

import pytest

from sourced_memory import PurgeResult, protect
from sourced_memory.llm import MockLLM
from sourced_memory.router import LLMRouter


def _pp_llm():
    """Force classification to personal_preference so trusted-source items
    become BELIEFs (and thus reach the underlying store)."""
    return MockLLM({"default": {"functional_type": "personal_preference",
                                "confidence": 0.9, "supported": True}})


class FakeMem0:
    """Duck-typed mem0.Memory with the real add-return shape."""
    def __init__(self):
        self.records: dict[str, dict] = {}
        self._next = 0
        self.delete_calls: list[str] = []
        self.delete_side_effect: list[Exception | None] | None = None

    def add(self, message, *, user_id=None, metadata=None, **kw):
        self._next += 1
        rid = f"id_{self._next}"
        self.records[rid] = {"message": message, "metadata": metadata}
        return {"results": [{"id": rid, "memory": message, "event": "ADD"}]}

    def delete(self, memory_id):
        self.delete_calls.append(memory_id)
        if self.delete_side_effect:
            err = self.delete_side_effect.pop(0)
            if err is not None:
                raise err
        self.records.pop(memory_id, None)


def test_purge_loops_delete_on_captured_ids(tmp_path):
    """Every BELIEF's captured id is passed to store.delete on purge."""
    mem0 = FakeMem0()
    memory = protect(
        mem0, trusted=["user"], untrusted=["web"],
        router=LLMRouter(_pp_llm()),
        audit_log_path=tmp_path / "audit.jsonl",
    )
    memory.user.session("s47").add("a")
    memory.user.session("s47").add("b")
    memory.user.session("s99").add("c")     # different session, must survive

    result = memory.purge(source_id="s47")
    assert isinstance(result, PurgeResult)
    assert result.records_deleted_from_backing_store == 2
    assert set(mem0.delete_calls) == {"id_1", "id_2"}
    assert "id_3" in mem0.records     # session s99 untouched
    assert result.has_gaps is False


def test_purge_partial_failure_is_reported_not_swallowed(tmp_path):
    """delete failing on some ids must surface in
    records_that_failed_backing_delete; NEVER silent."""
    mem0 = FakeMem0()
    memory = protect(
        mem0, trusted=["user"], untrusted=["web"],
        router=LLMRouter(_pp_llm()),
        audit_log_path=tmp_path / "audit.jsonl",
    )
    memory.user.session("s47").add("a")
    memory.user.session("s47").add("b")
    memory.user.session("s47").add("c")

    mem0.delete_side_effect = [None, RuntimeError("mem0 unavailable"), None]

    result = memory.purge(source_id="s47")
    assert result.records_deleted_from_backing_store == 2
    assert result.audit_entries_retained_for_retry == 1
    assert len(result.records_that_failed_backing_delete) == 1
    failed_id, err = result.records_that_failed_backing_delete[0]
    assert failed_id == "id_2"
    assert "RuntimeError" in err
    assert "mem0 unavailable" in err
    assert result.has_gaps is True


def test_purge_failed_delete_remains_retryable(tmp_path):
    """After a partial-failure purge, a second purge run retries the failed
    ids. Permanent orphan (record in mem0, not in audit log) is impossible
    as long as the id was captured on write."""
    mem0 = FakeMem0()
    memory = protect(
        mem0, trusted=["user"], untrusted=["web"],
        router=LLMRouter(_pp_llm()),
        audit_log_path=tmp_path / "audit.jsonl",
    )
    memory.user.session("s47").add("a")
    memory.user.session("s47").add("b")
    memory.user.session("s47").add("c")

    mem0.delete_side_effect = [None, RuntimeError("mem0 unavailable"), None]
    r1 = memory.purge(source_id="s47")
    assert r1.audit_entries_retained_for_retry == 1

    # Audit log still contains the failed record; it is retryable.
    remaining = [e for e in memory.audit() if e.source_id == "s47"]
    assert len(remaining) == 1
    assert remaining[0].backing_store_id == "id_2"

    # Second purge: mem0 is back. delete should succeed this time.
    mem0.delete_side_effect = [None]
    r2 = memory.purge(source_id="s47")
    assert r2.records_deleted_from_backing_store == 1
    assert r2.records_that_failed_backing_delete == []
    assert r2.has_gaps is False
    # No audit entries left for s47.
    assert not any(e.source_id == "s47" for e in memory.audit())
    # No records left in mem0 for s47 either.
    assert "id_2" not in mem0.records


def test_purge_unrecognized_add_return_marked_unreachable(tmp_path):
    """If store.add returned something we can not extract an id from, the
    record is counted under records_unreachable_in_backing_store, NOT
    silently under records_deleted."""
    class BareMem0:
        def __init__(self):
            self.added = []
        def add(self, message, **kwargs):
            self.added.append(message)
            return None      # no id, at all
        def delete(self, memory_id):
            raise AssertionError("should not be called")

    mem0 = BareMem0()
    memory = protect(
        mem0, trusted=["user"], untrusted=["web"],
        router=LLMRouter(_pp_llm()),
        audit_log_path=tmp_path / "audit.jsonl",
    )
    memory.user.session("s1").add("statement")

    result = memory.purge(source_id="s1")
    assert result.records_deleted_from_backing_store == 0
    assert result.records_unreachable_in_backing_store == 1
    assert result.has_gaps is True
    # The audit entry is still removed locally (no id to retry against),
    # and the unreachable count preserves the fact that mem0 may still
    # hold the record. This is the "we could not verify removal" state.


def test_purge_str_names_the_gap_when_things_failed(tmp_path):
    """str(PurgeResult) must make gaps loud, not treat them as success."""
    mem0 = FakeMem0()
    memory = protect(
        mem0, trusted=["user"], untrusted=["web"],
        router=LLMRouter(_pp_llm()),
        audit_log_path=tmp_path / "audit.jsonl",
    )
    memory.user.session("s1").add("a")
    mem0.delete_side_effect = [RuntimeError("nope")]

    result = memory.purge(source_id="s1")
    rendered = str(result)
    assert "COULD NOT VERIFY REMOVAL" in rendered
    assert "delete failed" in rendered
    assert "id_1" in rendered
    assert "may still exist" in rendered

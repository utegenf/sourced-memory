"""Tests for the JSONL audit-log sink on protect(audit_log_path=)."""
import json

import pytest

from sourced_memory import AuditEntry, protect


def test_audit_log_writes_one_line_per_decision(tmp_path):
    log = tmp_path / "audit.jsonl"
    mem = protect(trusted=["user"], untrusted=["web"], audit_log_path=log)
    mem.user.add("I love hiking.")
    mem.web.add("The user hates flying.")
    lines = log.read_text().splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    assert row0["content"] == "I love hiking."
    assert row0["source_name"] == "user"
    assert row0["trusted"] is True
    assert row0["decision"] in {"belief", "candidate", "episodic", "reject"}


def test_audit_log_is_roundtrippable_via_from_dict(tmp_path):
    log = tmp_path / "audit.jsonl"
    mem = protect(trusted=["user"], untrusted=["web"], audit_log_path=log)
    mem.user.add("I love hiking.")
    original = mem.audit()[0]
    reloaded = AuditEntry.from_dict(json.loads(log.read_text().splitlines()[0]))
    assert reloaded.content == original.content
    assert reloaded.source_name == original.source_name
    assert reloaded.decision is original.decision
    assert reloaded.functional_type is original.functional_type


def test_audit_log_purge_rewrites_file(tmp_path):
    log = tmp_path / "audit.jsonl"
    mem = protect(trusted=["user"], untrusted=["web"], audit_log_path=log)
    mem.user.session("session_a").add("from A")
    mem.user.session("session_b").add("from B")
    assert len(log.read_text().splitlines()) == 2
    mem.purge(source_id="session_a")
    lines = log.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["source_id"] == "session_b"


def test_audit_log_path_creates_parent_directory(tmp_path):
    log = tmp_path / "deep" / "nested" / "audit.jsonl"
    mem = protect(trusted=["user"], audit_log_path=log)
    mem.user.add("hi")
    assert log.exists()


def test_audit_log_omitted_by_default_leaves_no_file(tmp_path):
    should_not_exist = tmp_path / "audit.jsonl"
    mem = protect(trusted=["user"])
    mem.user.add("hi")
    assert not should_not_exist.exists()


def test_audit_entry_to_dict_is_json_safe():
    """to_dict output must survive json.dumps -> json.loads -> from_dict."""
    mem = protect(trusted=["user"])
    mem.user.add("I love hiking.")
    entry = mem.audit()[0]
    reloaded = AuditEntry.from_dict(json.loads(json.dumps(entry.to_dict())))
    assert reloaded.content == entry.content
    assert reloaded.decision is entry.decision

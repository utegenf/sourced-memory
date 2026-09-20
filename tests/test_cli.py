"""Tests for the `sourced-memory` CLI (inspect / decisions / purge)."""
import json

import pytest

from sourced_memory import protect
from sourced_memory.cli import main


def _make_log(tmp_path):
    log = tmp_path / "audit.jsonl"
    mem = protect(trusted=["user"], untrusted=["web"], audit_log_path=log)
    mem.user.add("I love hiking.")
    mem.web.add("The user hates flying.")
    mem.user.session("session_a").add("from A")
    mem.user.session("session_b").add("from B")
    return log


def test_inspect_summarizes_decisions(tmp_path, capsys):
    log = _make_log(tmp_path)
    rc = main(["inspect", str(log)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "total decisions : 4" in out
    assert "user" in out
    assert "web" in out


def test_inspect_on_empty_or_missing_file_is_graceful(tmp_path, capsys):
    missing = tmp_path / "does-not-exist.jsonl"
    rc = main(["inspect", str(missing)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no admission decisions" in out


def test_decisions_prints_every_entry(tmp_path, capsys):
    log = _make_log(tmp_path)
    rc = main(["decisions", str(log)])
    out = capsys.readouterr().out
    assert rc == 0
    # every observation should surface in the output as its rendered AuditEntry line
    for phrase in ("I love hiking.", "The user hates flying.", "from A", "from B"):
        assert phrase in out


def test_decisions_limit(tmp_path, capsys):
    log = _make_log(tmp_path)
    rc = main(["decisions", str(log), "--limit", "2"])
    out = capsys.readouterr().out
    assert rc == 0
    # last two lines were "from A" and "from B"
    assert "from A" in out
    assert "from B" in out
    assert "I love hiking." not in out


def test_purge_removes_matching_entries_and_rewrites_file(tmp_path, capsys):
    log = _make_log(tmp_path)
    rc = main(["purge", str(log), "--source", "session_a"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "purged 1" in out
    remaining = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    assert len(remaining) == 3
    for row in remaining:
        assert row.get("source_id") != "session_a"


def test_purge_no_matching_source_is_a_no_op(tmp_path, capsys):
    log = _make_log(tmp_path)
    original = log.read_text()
    rc = main(["purge", str(log), "--source", "does-not-exist"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Nothing to do" in out
    assert log.read_text() == original


def test_purge_missing_file_errors(tmp_path, capsys):
    missing = tmp_path / "does-not-exist.jsonl"
    rc = main(["purge", str(missing), "--source", "s1"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "no audit log to purge" in err

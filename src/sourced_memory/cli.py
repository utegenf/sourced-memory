"""``sourced-memory`` command-line interface.

Reads and manipulates the JSONL audit log produced by
:func:`sourced_memory.protect` when it is constructed with
``audit_log_path=``. Three subcommands:

    sourced-memory inspect  <path>                Summary + recent decisions
    sourced-memory decisions <path> [--limit N]   Every decision, in order
    sourced-memory purge <path> --source <id>     Drop entries by source_id

The CLI is deliberately thin: it does not open a live ``ProtectedMemory``,
it only reads and rewrites the audit log file. That keeps the operator
tool decoupled from the running agent, which is the point.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from .protect import AuditEntry


def _iter_entries(path: Path) -> Iterator[AuditEntry]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                print(
                    f"warning: {path}:{line_no} skipped (invalid JSON: {e})",
                    file=sys.stderr,
                )
                continue
            try:
                yield AuditEntry.from_dict(data)
            except (KeyError, ValueError) as e:
                print(
                    f"warning: {path}:{line_no} skipped (bad audit entry: {e})",
                    file=sys.stderr,
                )
                continue


def _fmt_entries(entries: Iterable[AuditEntry]) -> str:
    return "\n".join(str(e) for e in entries)


def _cmd_inspect(args: argparse.Namespace) -> int:
    path = Path(args.path)
    entries = list(_iter_entries(path))
    if not entries:
        print(f"{path}: no admission decisions recorded yet.")
        return 0

    counts = Counter(e.decision.value for e in entries)
    print(f"SOURCED MEMORY  ({path})")
    print(f"  total decisions : {len(entries)}")
    print(f"  belief    : {counts.get('belief', 0)}")
    print(f"  candidate : {counts.get('candidate', 0)}")
    print(f"  episodic  : {counts.get('episodic', 0)}")
    print(f"  rejected  : {counts.get('reject', 0)}")

    sources = Counter(e.source_name for e in entries)
    print()
    print("  decisions by source:")
    for name, n in sorted(sources.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {name:20s} {n}")

    limit = 10
    recent = entries[-limit:]
    print()
    print(f"  most recent {len(recent)} decisions:")
    for entry in recent:
        print(f"    {entry}")
    if len(entries) > limit:
        print(f"    ... ({len(entries) - limit} earlier)  ")
        print(f"    run `sourced-memory decisions {path}` for the full log.")
    return 0


def _cmd_decisions(args: argparse.Namespace) -> int:
    path = Path(args.path)
    entries = list(_iter_entries(path))
    if args.limit is not None and args.limit > 0:
        entries = entries[-args.limit:]
    if not entries:
        print(f"{path}: no admission decisions recorded yet.")
        return 0
    print(_fmt_entries(entries))
    return 0


def _cmd_purge(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"{path}: no audit log to purge.", file=sys.stderr)
        return 1
    kept = []
    removed = 0
    for entry in _iter_entries(path):
        if entry.source_id == args.source:
            removed += 1
        else:
            kept.append(entry)
    if removed == 0:
        print(f"{path}: no entries with source_id={args.source!r}. Nothing to do.")
        return 0
    with path.open("w", encoding="utf-8") as fh:
        for entry in kept:
            fh.write(json.dumps(entry.to_dict()) + "\n")
    print(f"purged {removed} entr{'y' if removed == 1 else 'ies'} with source_id={args.source!r}.")
    print(f"{path}: {len(kept)} entries remain.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sourced-memory",
        description="Inspect and remediate a sourced-memory audit log (JSONL).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser(
        "inspect",
        help="Summary of admission decisions in an audit log.",
    )
    p_inspect.add_argument("path", help="Path to the JSONL audit log.")
    p_inspect.set_defaults(func=_cmd_inspect)

    p_decisions = sub.add_parser(
        "decisions",
        help="Print every admission decision in the audit log.",
    )
    p_decisions.add_argument("path", help="Path to the JSONL audit log.")
    p_decisions.add_argument(
        "--limit", type=int, default=None,
        help="Show only the last N decisions. Default: all.",
    )
    p_decisions.set_defaults(func=_cmd_decisions)

    p_purge = sub.add_parser(
        "purge",
        help="Remove every audit entry with a given source_id.",
    )
    p_purge.add_argument("path", help="Path to the JSONL audit log.")
    p_purge.add_argument(
        "--source", required=True,
        help="source_id whose entries should be removed.",
    )
    p_purge.set_defaults(func=_cmd_purge)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

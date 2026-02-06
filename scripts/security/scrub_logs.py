#!/usr/bin/env python3
"""
Scrub API keys and bearer tokens from run/session logs.

Default scope is intentionally log-focused. Use --all-text to scan all text files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_diplomacy.redaction import redact_text


TEXT_EXTS = {
    ".log",
    ".txt",
    ".json",
    ".jsonl",
    ".csv",
    ".md",
    ".yaml",
    ".yml",
    ".tsv",
}

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
}

TARGET_PATH_PARTS = {
    "results",
    "runs",
    "logs",
    "codex_session_logs",
    "codex-chat-sessions",
    "analysis",
    "artifacts",
}

TARGET_FILENAMES = {
    "command.txt",
    "console.log",
    "general_game.log",
    "overview.jsonl",
    "llm_responses.csv",
    "runs_summary.json",
    "config.json",
}


def should_scan(path: Path, root: Path, all_text: bool) -> bool:
    if path.suffix.lower() not in TEXT_EXTS:
        return False
    rel = path.relative_to(root)
    rel_parts = {p.lower() for p in rel.parts}
    if rel_parts & SKIP_DIRS:
        return False
    if all_text:
        return True
    name_l = path.name.lower()
    if name_l in TARGET_FILENAMES:
        return True
    if "log" in name_l or "session" in name_l:
        return True
    return bool(rel_parts & TARGET_PATH_PARTS)


def scrub_file(path: Path, dry_run: bool) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    except Exception:
        return False

    redacted = redact_text(text)
    if redacted == text:
        return False

    if not dry_run:
        path.write_text(redacted, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Repo root to scan")
    parser.add_argument("--all-text", action="store_true", help="Scan all text files, not only log-like paths")
    parser.add_argument("--dry-run", action="store_true", help="Report files that would change without writing")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(str(root))

    scanned = 0
    changed = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if not should_scan(path, root, args.all_text):
            continue
        scanned += 1
        if scrub_file(path, args.dry_run):
            changed += 1

    result: Dict[str, object] = {
        "root": str(root),
        "scanned_files": scanned,
        "changed_files": changed,
        "dry_run": args.dry_run,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Exit-code guard when graphify-out/ is stale after phases/ or site/ edits.

Run after `graphify update .` on implementer sessions (see AGENTS.md § Graphify).
Use --strict before marking a graph-relevant task done. Stdlib only; never runs graphify.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRAPH_JSON = ROOT / "graphify-out" / "graph.json"
GRAPH_REPORT = ROOT / "graphify-out" / "GRAPH_REPORT.md"
BUILT_COMMIT_RE = re.compile(r"Built from commit: `([0-9a-f]{7,40})`")
QUIZ_JSON_RE = re.compile(r"^phases/[^/]+/[^/]+/quiz\.json$")
UPDATE_CMD = "py -3.12 -m graphify update ."


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=check,
    )


def _in_git_repo() -> bool:
    try:
        return _git("rev-parse", "--git-dir", check=False).returncode == 0
    except FileNotFoundError:
        return False


def _changed_paths() -> set[str]:
    paths: set[str] = set()
    for extra in ((), ("--cached",)):
        result = _git("diff", "--name-only", *extra, "HEAD", "--", "phases", "site", check=False)
        if result.returncode != 0:
            continue
        paths.update(line.strip() for line in result.stdout.splitlines() if line.strip())
    return paths


def _graph_relevant_changes(paths: set[str]) -> bool:
    """True when edits need graphify update (code/, site/, docs/, etc.)."""
    for raw in paths:
        path = raw.replace("\\", "/")
        if path.startswith("site/"):
            return True
        if QUIZ_JSON_RE.fullmatch(path):
            continue
        if path.startswith("phases/"):
            return True
    return False


def _resolve_commit(ref: str) -> str | None:
    result = _git("rev-parse", "--verify", f"{ref}^{{commit}}", check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _short(ref: str) -> str:
    result = _git("rev-parse", "--short", ref, check=False)
    if result.returncode == 0:
        return result.stdout.strip()
    return ref[:7]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check graphify-out/ freshness vs git HEAD after phases/ or site/ edits.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when the graph is stale (implementer gate)",
    )
    parser.add_argument(
        "--always",
        action="store_true",
        help="run the check even when phases/ and site/ are unchanged vs HEAD",
    )
    args = parser.parse_args(argv)

    if not _in_git_repo():
        print("skip: not a git repository")
        return 0

    if not args.always:
        changed = _changed_paths()
        if not changed:
            print("skip: no changes under phases/ or site/")
            return 0
        if not _graph_relevant_changes(changed):
            print("skip: only quiz.json changes under phases/ (no code/ or site/)")
            return 0

    if not GRAPH_JSON.is_file() or not GRAPH_REPORT.is_file():
        print(f"error: run: {UPDATE_CMD}")
        return 2

    report_text = GRAPH_REPORT.read_text(encoding="utf-8")
    match = BUILT_COMMIT_RE.search(report_text)
    if not match:
        print("error: cannot parse built commit from GRAPH_REPORT.md")
        return 2

    built_ref = match.group(1)
    built_full = _resolve_commit(built_ref)
    if built_full is None:
        print("error: cannot parse built commit from GRAPH_REPORT.md")
        return 2

    head_result = _git("rev-parse", "HEAD", check=False)
    if head_result.returncode != 0:
        print("error: cannot parse built commit from GRAPH_REPORT.md")
        return 2
    head_full = head_result.stdout.strip()

    if built_full == head_full:
        print(f"ok: graph built at {_short(built_full)} matches HEAD")
        return 0

    print(
        f"stale: graph built at {_short(built_full)}, HEAD is {_short(head_full)}.\n"
        f"     Run: {UPDATE_CMD}"
    )
    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())

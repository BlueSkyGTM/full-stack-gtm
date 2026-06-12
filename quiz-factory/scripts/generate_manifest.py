#!/usr/bin/env python3
"""Build quiz-factory/manifest.json from current repo state.

Usage (repo root):
    python quiz-factory/scripts/generate_manifest.py
    python quiz-factory/scripts/generate_manifest.py --write
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASES = ROOT / "phases"
OUT = ROOT / "quiz-factory" / "manifest.json"

PHASE_ORDER = ["07", "08", "10", "06", "09", "12", "13", "15", "16", "17", "18"]
LESSON_RE = re.compile(r"^[0-9]{2}-[a-z0-9][a-z0-9-]*[a-z0-9]$")
PHASE_RE = re.compile(r"^([0-9]{2})-")

# Quizzes generated as the all-"A" anti-pattern (every correct=0). Positions were
# mechanically balanced, but the content is unverified — Claude Code must rebuild
# them from docs/en.md + code/ with the variance rule. See REFERENCES.md "Anti-pattern".
REDO_QUIZ = {
    "phases/07-transformers-deep-dive/15-attention-variants",
    "phases/07-transformers-deep-dive/16-speculative-decoding",
    "phases/08-generative-ai/19-visual-autoregressive-var",
    "phases/10-llms-from-scratch/15-speculative-decoding-eagle3",
    "phases/10-llms-from-scratch/16-differential-attention-v2",
    "phases/10-llms-from-scratch/17-native-sparse-attention",
    "phases/10-llms-from-scratch/34-gradient-checkpointing",
}
REDO_NOTE = "all-A anti-pattern; rebuild questions + variance from docs/en.md and code/"

# Schema-repair phases — discovered by 2026-05-30 QA audit.
# Every quiz in these phases has the wrong stage count/sequence.
# key = two-digit phase number
# value = (wrong_stage_tuple, repair_note)
SCHEMA_REPAIR_PHASES: dict[str, tuple[tuple[str, ...], str]] = {
    "04": (
        ("pre", "pre", "post", "post", "post"),
        "5q (pre×2, post×3) → rebuild to (pre×1, check×3, post×2): "
        "keep best pre; write 3 new check questions from distinct doc sections; "
        "keep 2 posts; for Build lessons tie at least one check to a named function in code/main.*",
    ),
    "05": (
        ("pre", "pre", "check", "check", "check", "post", "post", "post"),
        "8q (pre×2, check×3, post×3) → trim to (pre×1, check×3, post×2): "
        "drop the weaker pre; drop the weakest post OR merge two posts into one "
        "integration question that requires two doc ideas; keep all three checks unchanged",
    ),
    "14": (
        ("pre", "pre", "check", "check", "check", "post", "post"),
        "7q (pre×2, check×3, post×2) → trim to (pre×1, check×3, post×2): "
        "drop the weaker pre; add a code-symbol reference (exact function name from "
        "code/main.*) to at least one check or post",
    ),
}


def lesson_path(phase: Path, lesson: Path) -> str:
    return f"phases/{phase.name}/{lesson.name}"


def _read_questions(lesson: Path) -> list[dict]:
    quiz = lesson / "quiz.json"
    if not quiz.is_file():
        return []
    data = json.loads(quiz.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("questions", [])
    return []


def schema_repair_note(phase_num: str, lesson: Path) -> str | None:
    """Return the repair note if this lesson needs schema_repair, else None."""
    if phase_num not in SCHEMA_REPAIR_PHASES:
        return None
    expected_pattern, note = SCHEMA_REPAIR_PHASES[phase_num]
    questions = _read_questions(lesson)
    if not questions:
        return None
    actual = tuple(q.get("stage", "") for q in questions)
    if actual == expected_pattern:
        return note
    return None


def job_for_lesson(lesson: Path) -> str | None:
    quiz = lesson / "quiz.json"
    if not quiz.is_file():
        return "create_quiz"
    questions = _read_questions(lesson)
    for q in questions:
        if not str(q.get("explanation", "")).strip():
            return "fill_explanations"
    return None


def iter_lessons():
    for phase in sorted(PHASES.iterdir()):
        if not phase.is_dir():
            continue
        m = PHASE_RE.match(phase.name)
        if not m:
            continue
        phase_num = m.group(1)
        for lesson in sorted(phase.iterdir()):
            if lesson.is_dir() and LESSON_RE.match(lesson.name):
                yield phase_num, phase, lesson


def build_manifest() -> dict:
    rows = []
    for phase_num, phase, lesson in iter_lessons():
        path = lesson_path(phase, lesson)
        if path in REDO_QUIZ:
            job, note = "redo_quiz", REDO_NOTE
        else:
            repair_note = schema_repair_note(phase_num, lesson)
            if repair_note is not None:
                job, note = "schema_repair", repair_note
            else:
                job, note = job_for_lesson(lesson), ""
        if job is None:
            continue
        rows.append(
            {
                "path": path,
                "phase": phase_num,
                "job_type": job,
                "status": "pending",
                "note": note,
            }
        )

    # Priority order: redo_quiz (0) → schema_repair (1) → everything else (2)
    # Within each priority: phase order, then path.
    job_rank = {"redo_quiz": 0, "schema_repair": 1}
    schema_repair_phase_order = sorted(SCHEMA_REPAIR_PHASES.keys())

    def sort_key(row: dict) -> tuple:
        if row["job_type"] == "schema_repair":
            try:
                order = schema_repair_phase_order.index(row["phase"])
            except ValueError:
                order = 99
        else:
            try:
                order = PHASE_ORDER.index(row["phase"])
            except ValueError:
                order = 99
        return (job_rank.get(row["job_type"], 2), order, row["path"])

    rows.sort(key=sort_key)
    create = sum(1 for r in rows if r["job_type"] == "create_quiz")
    fill = sum(1 for r in rows if r["job_type"] == "fill_explanations")
    redo = sum(1 for r in rows if r["job_type"] == "redo_quiz")
    repair = sum(1 for r in rows if r["job_type"] == "schema_repair")
    return {
        "version": 1,
        "phase_order": PHASE_ORDER,
        "summary": {
            "pending_rows": len(rows),
            "redo_quiz": redo,
            "schema_repair": repair,
            "create_quiz": create,
            "fill_explanations": fill,
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"write {OUT.relative_to(ROOT)}",
    )
    args = parser.parse_args()
    manifest = build_manifest()
    text = json.dumps(manifest, indent=2) + "\n"
    if args.write:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)}")
    else:
        print(text)
    s = manifest["summary"]
    print(
        f"# pending={s['pending_rows']} redo_quiz={s['redo_quiz']} "
        f"schema_repair={s['schema_repair']} create_quiz={s['create_quiz']} "
        f"fill_explanations={s['fill_explanations']}",
        file=__import__("sys").stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

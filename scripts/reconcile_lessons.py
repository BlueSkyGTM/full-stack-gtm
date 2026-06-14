"""
reconcile_lessons.py — disk-truth reconciliation for the lean fundamentals cut.

The manifest says every lesson is "done". It lies — "done" only meant a file
got written. This script reads the actual content of all 510 lessons, classifies
each against the lean-fundamentals cut line, rewrites the manifest status
truthfully, and harvests each lesson's deep back-half (Ship It) into the Loop 2
seed bank.

It does NOT mutate lesson content. Classification + manifest rewrite + harvest
copy only. Fully reversible (commit the 510 state before running).

Buckets (cut line = keep Concept + Build It + runnable weave; defer heavy Ship It):
  lean-ready  : complete (## Sources + balanced fences) -> manifest status "done"
  close       : truncated in/after Use It; front half good -> status "failed", bucket "close"
  complete    : died during Build It -> status "failed", bucket "complete"
  regen       : died before Build It -> status "failed", bucket "regen"

Usage:
  python scripts/reconcile_lessons.py --dry-run     # report only, no writes
  python scripts/reconcile_lessons.py               # rewrite manifest + harvest seed bank
"""

import json
import re
import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "stages/02-lesson-injection/output/manifest.json"
SEED_ROOT = REPO / "stages/02-lesson-injection/output/intermediate-seed"


def classify(text: str) -> str:
    has_build = "## Build It" in text
    has_use = "## Use It" in text
    has_src = "## Sources" in text
    balanced = text.count("```") % 2 == 0
    if has_src and balanced:
        return "lean-ready"
    if has_build and has_use:
        return "close"
    if has_build:
        return "complete"
    return "regen"


def extract_section(text: str, header: str) -> str | None:
    """Return the `## {header}` section through the next `## ` (exclusive), or None."""
    m = re.search(
        r"(^##\s+" + re.escape(header) + r"\b.*?)(?=^##\s+|\Z)",
        text, re.MULTILINE | re.DOTALL,
    )
    return m.group(1).strip() if m else None


def main():
    ap = argparse.ArgumentParser(description="Disk-truth reconciliation for the lean cut")
    ap.add_argument("--dry-run", action="store_true", help="report only, no writes")
    args = ap.parse_args()

    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    counts = {"lean-ready": 0, "close": 0, "complete": 0, "regen": 0, "missing": 0}
    harvested = 0

    for row in rows:
        rel = (row.get("output_file") or "").replace("\\", "/").strip()
        out = REPO / rel if rel else None
        if not rel or out is None or not out.is_file():
            counts["missing"] += 1
            row["status"] = "failed"
            row["bucket"] = "regen"
            continue

        text = out.read_text(encoding="utf-8")
        bucket = classify(text)
        counts[bucket] += 1

        if bucket == "lean-ready":
            row["status"] = "done"
            row["bucket"] = "lean-ready"
        else:
            row["status"] = "failed"
            row["bucket"] = bucket

        # Harvest the deep back-half (Ship It) into the Loop 2 seed bank.
        ship = extract_section(text, "Ship It")
        if ship and not args.dry_run:
            seed = SEED_ROOT / row["phase"] / row["lesson"]
            seed.mkdir(parents=True, exist_ok=True)
            (seed / "ship-it.md").write_text(ship, encoding="utf-8")
            harvested += 1

    total = len(rows)
    print(f"Reconciled {total} lessons:")
    for k in ("lean-ready", "close", "complete", "regen", "missing"):
        print(f"  {k:11}: {counts[k]:3}  ({100*counts[k]//total}%)")
    needs_work = counts["close"] + counts["complete"] + counts["regen"] + counts["missing"]
    print(f"\n  status=done (lean-ready): {counts['lean-ready']}")
    print(f"  status=failed (needs work): {needs_work}")

    if args.dry_run:
        print("\n[dry-run] manifest NOT modified, seed bank NOT written.")
        return

    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(MANIFEST)
    print(f"\nManifest rewritten with true status + bucket per row.")
    print(f"Seed bank: harvested {harvested} Ship-It sections to {SEED_ROOT.relative_to(REPO)}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Query the local code graph. Usage: python3 scripts/query_graph.py query "question"."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        print("usage: query_graph.py <query|path|explain> ...", file=sys.stderr)
        return 2
    result = subprocess.run(
        ["py", "-3.12", "-m", "graphify", *args],
        cwd=ROOT,
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

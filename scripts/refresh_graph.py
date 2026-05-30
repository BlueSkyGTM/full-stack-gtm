#!/usr/bin/env python3
"""Rebuild the local code graph (AST, no LLM). See .cursor/skills/refresh/SKILL.md."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    result = subprocess.run(
        ["py", "-3.12", "-m", "graphify", "update", "."],
        cwd=ROOT,
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

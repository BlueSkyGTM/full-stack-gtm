#!/usr/bin/env python3
"""Check local code graph freshness vs git HEAD. Wraps check_graphify_freshness.py."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "check_graphify_freshness.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")

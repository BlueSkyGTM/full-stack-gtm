## Ship It

Wrap the detector in a CLI tool that reads from stdin and exits with a status code. This makes it usable as a pre-processing step in shell pipelines, CI checks, or any script that chains commands.

```python
#!/usr/bin/env python3
import sys
import json


def check(input_text, threshold=0.5):
    verdict = three_signal_detect(input_text, threshold=threshold)
    is_flagged = verdict["confidence"] >= threshold
    return {
        "input": input_text[:200],
        "verdict": "FLAGGED" if is_flagged else "SAFE",
        "confidence": round(verdict["confidence"], 3),
        "category": verdict["category"],
        "signal_count": len(verdict["signals"]),
        "signals": [
            {"rule": s["rule"], "category": s["category"], "
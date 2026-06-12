# Copy-Paste Flag Format
<!-- Stage 00-d output | 2026-06-12 -->
<!-- STATUS: LOCKED — exact string format. No changes without rebuilding all Stage 05 exercises. -->
<!-- Lock confirmation: Reviewed and locked on 2026-06-12 during Stage 00-d. -->

## The Flag

When a student completes a copy-paste exercise, they paste the terminal output into the Helix chat. Helix parses the output for this exact flag:

```
BLUESKYGTM_CHECK: OK
```

That is the canonical flag. Capital letters. Colon. Space. OK. No trailing characters, no version number, no phase prefix.

## Full Flag Specification

| Property | Value |
|----------|-------|
| Exact string | `BLUESKYGTM_CHECK: OK` |
| Case | All uppercase — `blueskygtm_check: ok` does NOT match |
| Whitespace | Exactly one space after the colon |
| Position | Anywhere in the pasted output — not required on its own line |
| Trailing content | Allowed — the parser looks for substring presence, not exact line match |
| ANSI codes | Stripped before matching — color escape sequences do not affect detection |
| CRLF | Normalized to LF before matching |
| Multiple occurrences | First match wins |

## How Exercises Emit the Flag

Every copy-paste exercise script must `print` or `echo` the flag as the final line of successful output:

```python
# Python
print("BLUESKYGTM_CHECK: OK")
```

```bash
# Bash
echo "BLUESKYGTM_CHECK: OK"
```

The flag is emitted only on success. On failure, the script must exit with a non-zero code and NOT emit the flag. Emitting the flag on a failed run defeats the purpose of the exercise check.

## How Helix Parses It

Helix's flag parser (implementation in Stage 05):

```python
import re

def check_flag(student_output: str) -> bool:
    # Strip ANSI escape codes
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean = ansi_escape.sub('', student_output)
    # Normalize line endings
    clean = clean.replace('\r\n', '\n').replace('\r', '\n')
    # Collapse multiple spaces (not inside quoted strings — simple approach)
    return 'BLUESKYGTM_CHECK: OK' in clean
```

If the flag is not found after normalization, Helix falls through to ASSESS — it does not display an error or break the student's flow. The student sees a normal response asking what they're working on.

## What the Flag Does NOT Do

- It does not mark a lesson complete. Lesson completion is a student action via the site's Mark Complete button.
- It does not trigger FSRS scheduling. Copy-paste exercises are evaluated, not recalled.
- It does not validate the correctness of the student's approach — only that they ran the exercise and got a successful exit. Deeper correctness is evaluated by Helix after seeing the flag.

## Failure Mode: False Positives

If a student manufactures the flag without running the exercise (`echo "BLUESKYGTM_CHECK: OK"`), Helix will accept it. This is a known and acceptable tradeoff. The course is self-paced and self-directed — there is no proctored enforcement. Students who skip exercises skip their own learning.

Helix's response after detecting the flag asks a follow-up question about what the student observed. A student who faked the flag will fail to answer the follow-up coherently. Helix moves them to HINT or CORRECT based on their response — the flag detection is not the final gate.

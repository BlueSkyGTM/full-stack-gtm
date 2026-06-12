# Anti-Patterns

Design mistakes to avoid before enabling unattended loops.

## 1. Same agent implements and verifies
One session marks its own work "done." Do instead: separate verifier sub-agent. Verifier default stance: REJECT.

## 2. No attempt cap
"Keep trying until CI is green." Do instead: hard cap (3 attempts) → escalate with full context in state file.

## 3. Vague triage output
Triage skill returns paragraphs of narrative. Do instead: structured markdown sections with one-line items and explicit `Suggested loop action`.

## 4. L3 before L1 quality
Auto-fix and auto-PR on day one. Do instead: L1 report-only week one. Measure triage accuracy before enabling L2.

## 5. Shared state without schema
Three loops append to one unstructured STATE.md. Do instead: one state file per pattern, or clearly separated sections with prune rules.

## 6. MCP with write-everything scope
Loop can merge PRs, post to Slack, edit production tickets on day one. Do instead: L1 read-only connectors. Expand scope only after trust is earned.

## 7. No kill switch
Loop runs 24/7 with no pause criteria. Do instead: document pause/kill in LOOP.md.

## 8. Fixing flakes with code
CI Sweeper changes application code when classification is `flake`. Do instead: classify → quarantine or retry policy → escalate.

## 9. Auto-merge without allowlist
"Verifier passed, merge it." Do instead: explicit path allowlist; human merge for denylist paths.

## 10. No run log
Only STATE.md, no history of what the loop did. Do instead: append to `loop-run-log.md` every run.

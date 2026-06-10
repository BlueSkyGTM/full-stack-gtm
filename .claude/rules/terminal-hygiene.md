---
description: Terminal hygiene for Cline — backend execution shell, not only the Editor panel
alwaysApply: true
---

# Terminal hygiene (Cline)

Two different terminals matter. **Do not only clean the Editor Terminal panel** (`View → Terminal`).

| Surface | What it is | Symptom when stale |
|---------|------------|-------------------|
| **Editor terminals** | Tabs in the bottom Terminal panel | Clutter; sometimes contributes to VS Code shell-integration limits |
| **Cline execution terminal** | Backend shell Cline uses for `execute_command` (standalone/background runtime) | Submit greyed out, soft lock, Proceed never advances — **even when the Editor panel looks fine** |

Confirmed fix path: hygiene on the **Cline** side first, then Editor terminals if needed.

## Cline execution terminal (check this first)

When submit is disabled or Cline is stuck after a command:

1. **Proceed** or **Cancel** in the Cline panel for the hung step.
2. Open the **command output / terminal area inside the Cline task** (where the last `execute_command` ran). If a shell is still running, stop it (**Ctrl+C** in that context, or Cancel the step).
3. **Cline Settings** (gear) → **Terminal** → prefer **Background Exec** / standalone mode so completion does not depend on Editor shell integration.
4. Start a **new task** (`+` or `/newtask`) if the backend shell is wedged — same window, fresh runtime.

## Editor Terminal panel (secondary)

- Prefer one compound command over many sequential `execute_command` calls.
- Batch git: `git add A && git commit -m "..."`.
- Avoid bare `git status` / `git log` — use `--short`, `--oneline -N`, or pipe output.
- On Windows PowerShell: prefer `Remove-Item` with comma-separated paths over ambiguous `del`/`rm`.
- If Editor tab count is high: kill extras (trash icon) or **Terminal: Kill All Terminals** from the palette — after fixing the Cline task shell above.

## When it still fails

`Ctrl+Shift+P` → **Developer: Reload Window**. Save editors first.

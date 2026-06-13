# Lesson Outline: Claude Code as an Autonomous Agent — Permission Modes and Auto Mode

---

## Beat 1: Hook

You built a batch script that processes 500 accounts through an enrichment waterfall. Claude Code pauses for permission on every file write. You approve the first 3. You approve the 10th. You stop approving after the 40th and the job dies halfway through. Permission modes exist to solve exactly this — trading oversight granularity for execution continuity.

---

## Beat 2: Concept

**The mechanism: a delegation spectrum.** Every agent operation falls on a scale from "ask every time" to "execute autonomously within bounds." Claude Code implements this through discrete permission modes that control which operations require human confirmation — file reads, file writes, shell command execution, and network calls. The default mode is conservative: write operations and shell commands pause for approval. Auto mode relaxes these constraints by whitelisting operation categories. The tradeoff is speed vs. containment — auto mode is faster but narrows your intervention window if the agent drifts.

**Key distinctions:**
- **Interactive mode**: every mutation (write, execute) requires explicit approval. Safe. Slow for batch.
- **Auto-accept edits**: file edits within the working directory proceed without confirmation. Shell commands still pause.
- **Full auto mode (`--dangerously-skip-permissions`)**: all operations execute. Designed for CI/CD and containerized environments, not local dev machines with production credentials.

The permission model is not a binary — it is a configurable boundary. Understanding where the boundary sits determines whether autonomous execution is safe for a given task context.

---

## Beat 3: Demo

**Show the three modes with observable output.**

- **Demo A**: Run a simple file creation + shell command in default interactive mode. Show the permission prompts (or mock them in script output). Time the execution.
- **Demo B**: Same operation sequence with auto-accept on edits. Show which prompts disappear, which remain. Time delta.
- **Demo C**: Same operation in full auto mode. Show uninterrupted execution. Note the `--dangerously-skip-permissions` flag name — the naming itself communicates the risk posture.

Each demo produces a timestamped log showing wall-clock time and permission checkpoint count.

---

## Beat 4: Use It — GTM Redirect

**GTM cluster: Zone B — Enrichment & Research Automation (Batch Processing)**

[CITATION NEEDED — concept: Clay waterfall permission automation for batch enrichment jobs]

The GTM application: when you run a multi-step enrichment pipeline (fetch company data → score against ICP → write results to CSV → append to CRM), the pipeline is deterministic by step 3. The agent is not making novel decisions — it is executing a known sequence. Auto mode is appropriate here because the operations are predictable and the blast radius is limited to a staging directory. The redirect: batch enrichment jobs in GTM are the canonical use case for relaxed permission modes. If your Clay waterfall writes 500 rows, you do not approve 500 writes. The same principle applies to autonomous agents.

**Exercise hooks:**
- *Easy*: Configure auto-accept for a single file enrichment script. Confirm it processes 10 records without pausing.
- *Medium*: Build a two-stage pipeline (fetch + write) that runs unattended in a container. Log each stage completion with timestamps.
- *Hard*: Implement a guard condition — a pre-execution check that refuses auto mode if the target directory contains files matching production naming patterns (e.g., `prod_*`, `live_*`).

---

## Beat 5: Ship It

Ship a self-contained batch processor that runs in auto mode inside a disposable directory structure. The processor reads a list of domains from a file, simulates an enrichment step (or calls a real API if credentials exist), writes one output file per domain, and produces a summary log. The entire run completes without human interaction. The deliverable is the processor script, the directory structure it operates on, and the summary log output.

**Constraint**: the script must refuse to execute if it detects it is running outside a `workspace/` directory. This demonstrates that permission relaxation requires boundary enforcement elsewhere.

---

## Beat 6: Evaluate

**5 questions, grounded in observable behavior:**

1. Which operation category still requires explicit approval in auto-accept-edits mode?
2. What does the `--dangerously-skip-permissions` flag name signal about its intended use case?
3. Given a 4-step pipeline (read → transform → write → shell command), predict which steps will pause for permission under each mode.
4. A batch job in auto mode writes to the wrong directory. Identify the missing guard condition.
5. Compare: when is full auto mode safer than interactive mode? (Hint: CI/CD containers with no production access vs. local dev machines.)

---

## Learning Objectives

1. Configure Claude Code permission modes to match task autonomy requirements.
2. Compare the operation categories each permission mode whitelists.
3. Implement a multi-step file operation that completes without manual intervention.
4. Detect unsafe contexts for auto mode by evaluating blast radius and credential exposure.
5. Evaluate when permission relaxation is appropriate for batch GTM data processing.

---

## GTM Redirect Rules (summary)

- **Beat 4 (Use It)**: Zone B — enrichment batch processing. The redirect is specific: autonomous agents running predictable multi-step pipelines (fetch → score → write) are the GTM analog of Clay waterfall execution. You do not approve every row.
- **Beat 5 (Ship It)**: foundational for Zone B. The guard condition pattern (refuse execution outside safe boundaries) transfers directly to any autonomous GTM workflow.
- If the AI concept does not cleanly map deeper, the redirect holds at "foundational for Zone B batch automation." No fabrication.
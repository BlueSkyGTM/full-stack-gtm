# LOOP.md — Loop Engineering Standards for the Full-Stack GTM Build Pipeline

> "My job is to write loops." — Boris Cherny, Head of Claude Code, Anthropic

This file defines the loop engineering standards for the build pipeline that produces the Full-Stack GTM course. It maps the five loop primitives to our specific build context and establishes the operating model before any Stage runs.

Loop engineering is not a new tool. It is the framework that was already implicit in how this pipeline works — batch-orchestration, maker/checker via autoplan, CONTEXT.md as skills, vault as state. This document makes the pattern explicit and standard.

---

## Our Five Primitives

| Primitive | Loop Engineering Definition | This Pipeline |
|-----------|---------------------------|---------------|
| **Scheduling / Automation** | Discovery + triage on a cadence | Manual trigger per Stage. Human initiates each Stage; the loop does not auto-advance. Cadence is human-gated by design (File-to-Pipe). |
| **Worktrees** | Safe parallel execution | batch-orchestration skill: each batch round runs up to 8 subagents in isolated parallel tracks. Each subagent gets a scoped manifest slice — never the whole queue. |
| **Skills** | Persistent project knowledge | CONTEXT.md files are the skills. Every Stage reads its CONTEXT.md before executing. The vault (variable-registry, course-identity, helix-voice) is the shared skill layer across all Stages. |
| **Plugins / Connectors** | Reach into real tools via MCP | gstack skills (autoplan, review, ship) + GLM air (Newton gap-fill research). MCP wires Claude Code to n8n, NocoDB, and the GTM warehouse in the revenue pipeline. |
| **Sub-agents (Maker/Checker)** | The agent that builds ≠ the agent that verifies | Lyra builds (Stages 01-04). Hypatia audits (Stage 09). autoplan is the CEO/Eng/DX checker on every plan. These are never the same agent run. |
| **Memory / State** | Durable spine outside any conversation | `vault/` = course identity state. `pipeline_runs` table = GTM build audit log. gstack brain = host cross-session memory. The manifest (Stage 01 output) = resumable queue state — the pipeline never restarts from zero. |

---

## Loop Anatomy for a Build Stage

Every Stage follows this loop anatomy:

```
Schedule (human trigger)
  → Read CONTEXT.md (Triage Skill)
  → Check STATE (vault/, manifest, prior stage outputs)
  → Isolated batch run (Worktree — batch-orchestration)
  → Maker executes (Lyra / Claude Code)
  → Checker verifies (audit checks in CONTEXT.md, autoplan for plans)
  → Human Gate? (if stage has a gate: STOP. If not: continue.)
  → Write outputs → Update STATE
  → Notify (gstack review log, MCP notification)
  → Loop closes (next Stage opens on next human trigger)
```

---

## Phased Rollout (L1 → L2 → L3)

The build pipeline follows the standard loop rollout:

| Level | What It Means Here | Current Status |
|-------|-------------------|----------------|
| **L1 — Report** | Claude Code reads and reports. No writes. Used during 00-a archaeology and 00-b content mapping. | Active in Phase 0 |
| **L2 — Assisted** | Claude Code writes with human review before commit. All Stage 01-09 batch work runs at this level. Human approves each round. | Active for Stages 01-09 |
| **L3 — Unattended** | Loop runs without per-action human approval. Not used in this pipeline until Stage 10 validation is proven. | Deferred — only after Stage 10 |

**Default is L2.** Stage 10 is the gate to L3. No Stage runs unattended before Stage 10 validation passes.

---

## STATE.md Pattern

Each Stage that produces durable outputs must maintain a STATE entry. The manifest (produced at Stage 01) is the primary state file. For Stages that predate the manifest:

| Stage | State Location | What It Tracks |
|-------|---------------|----------------|
| 00-a through 00-e | `vault/variable-registry.md` | All resolved variables; current vault completeness |
| Stage 01+ | `stages/01-gtm-skeleton/output/manifest.json` | 498-lesson queue, each lesson's status (pending / drafted / audited / blocked) |
| GTM pipeline | `pipeline_runs` table | Every n8n Stage run, status, timestamp, what it touched |

If a loop run produces no STATE update, it is not a complete loop — it is a prompt.

---

## Maker / Checker Assignments

| Stage | Maker | Checker |
|-------|-------|---------|
| 00-a through 00-e | Claude Code | autoplan (plan review before execution) |
| Stage 01 (outlines) | Lyra (content) | Claude Code spot-checks 5 outlines per phase |
| Stage 02 (lesson injection) | Lyra (content) | Derivation gate (chain-of-reason check before prose) |
| Stage 03 (exercises) | Lyra (content) | Claude Code verifies CLI-completability of 10% sample |
| Stage 04 (recall cards) | Lyra (content) | FSRS format audit (schema validation) |
| Stage 05 (Helix build) | Lyra (code) | 3-scenario test harness (required before gate) |
| Stage 06 (site readability) | Lyra (code) | design-review skill + visual diff |
| Stage 09 (quality pass) | Hypatia | CLARITY + WEAVE + ACCURACY judges (numeric thresholds) |
| Stage 10 (validation) | Claude Code | Human — no automated checker for the final gate |

---

## Human Gates in the Loop

Gates are not failures — they are the loop deciding it cannot proceed without human judgment. Every gate must provide the human with full context: what the loop did, what it found, and what the two or three possible decisions are.

| Gate | Trigger Condition | Human Decision Required |
|------|-----------------|------------------------|
| Phase 0 Gate 1 | After 00-d: student-state-options.md produced | Choose the state mechanism before 00-e-full runs |
| Phase 0 Gate 2 | After 00-e-full: vault complete | Approve helix-voice.md and student promise before Stage 01 |
| Stage 01 dry-run | Before full Stage 01 batch | Single lesson outline passes structure check |
| Stage 06 | Visual diff of site changes | Approve or reject rendering changes |
| Stage 09 | Quality judge results | Unblock lessons flagged by CLARITY/WEAVE/ACCURACY |
| Stage 10 | Full validation run | Ship decision — human only, no automation |

---

## Anti-Patterns We Are Explicitly Avoiding

From the loop-engineering framework:

| Anti-Pattern | How We Avoid It |
|-------------|-----------------|
| **Cognitive surrender** — loop runs without a human reading what it ships | L2 default: human reviews every batch round before commit |
| **No STATE** — loop has no memory, re-derives everything on each run | vault/ + manifest + pipeline_runs are always written before the loop closes |
| **No verifier** — maker and checker are the same agent | Maker/checker table above: they are never the same run |
| **Premature L3** — going unattended before the loop is proven | Stage 10 is the L3 gate. No exceptions. |
| **Giant batches** — sending everything to one agent | batch-orchestration caps at 8 parallel subagents per round |

---

## References

- Loop Engineering framework: `references/loop-engineering/` (primitives, patterns, checklist)
- batch-orchestration skill: `.claude/skills/batch-orchestration/SKILL.md`
- Manifest pattern: `stages/01-gtm-skeleton/CONTEXT.md`
- GTM loop architecture: `CLAUDE.md` (the other repo — File-to-Pipe Engineering)

# Curriculum QA Findings — 2026-05-30

**Audit method:** 5 parallel subagents, each reading `docs/en.md` + `quiz.json` for 4 sampled lessons, scoring against the 18-criterion rubric in `quiz-factory/QUALITY-STANDARDS.md`. Subagents A–D reported scores out of 32 (as instructed); Subagent E self-computed 36. All scores are normalized to % for cross-comparison.

**Sample:** 20 lessons across 5 risk-stratified tiers.

---

## Per-lesson grade table

### Tier A — Gold baseline (rebuilt 2026-05-30)

| Lesson | Raw | / max | % | Grade | P0 |
|--------|-----|-------|---|-------|-----|
| `07/15-attention-variants` | 31 | 32 | 97% | **A** *(P0 fixed, see below)* | ~~check Q3 wrong answer~~ FIXED |
| `07/16-speculative-decoding` | 32 | 32 | 100% | **A** | none |
| `10/15-speculative-decoding-eagle3` | 31 | 32 | 97% | **A** | none |
| `10/34-gradient-checkpointing` | 32 | 32 | 100% | **A** | none |

**Tier pass rate (B+):** 4/4 = **100%** (after fix)

---

### Tier B — Phase 04 computer-vision (complete phase, older)

| Lesson | Raw | / max | % | Grade | P0 |
|--------|-----|-------|---|-------|-----|
| `04/02-convolutions-from-scratch` | 29 | 32 | 91% | **C** (capped) | C1: 5q, wrong stages |
| `04/07-semantic-segmentation-unet` | 28 | 32 | 88% | **C** (capped) | C1: 5q, wrong stages |
| `04/14-vision-transformers` | 29 | 32 | 91% | **C** (capped) | C1: 5q, wrong stages |
| `04/22-3d-gaussian-splatting` | 30 | 32 | 94% | **C** (capped) | C1: 5q, wrong stages |

**Bonus check — `04/04-image-classification`:** confirmed same P0 pattern (5q, `pre×2, post×3`, no `check` stage).

**Tier pass rate (B+):** 0/4 = **0%** (content quality high; blocker is structural C1)

---

### Tier C — Phase 05 NLP foundations-to-advanced (complete phase, older)

| Lesson | Raw | / max | % | Grade | P0 |
|--------|-----|-------|---|-------|-----|
| `05/03-word-embeddings-word2vec` | 30 | 32 | 94% | **C** (capped) | C1: 7q, `pre×2, check×3, post×3` |
| `05/08-cnns-rnns-for-text` | 31 | 32 | 97% | **C** (capped) | C1: 7q, wrong stages |
| `05/19-subword-tokenization` | 30 | 32 | 94% | **C** (capped) | C1: 7q, wrong stages |
| `05/27-llm-evaluation-frameworks` | 27 | 32 | 84% | **C** (capped) | C1: 7q, wrong stages |

**Tier pass rate (B+):** 0/4 = **0%** (same pattern — content strong, structure wrong)

---

### Tier D — Phase 14 agent-engineering (large, varied)

| Lesson | Raw | / max | % | Grade | P0 |
|--------|-----|-------|---|-------|-----|
| `14/01-the-agent-loop` | 30 | 32 | 94% | **C** (capped) | C1: 7q, `pre×2` |
| `14/12-anthropic-workflow-patterns` | 29 | 32 | 91% | **C** (capped) | C1: 7q, `pre×2` |
| `14/26-failure-modes-agentic` | 32 | 32 | 100% | **C** (capped) | C1: 7q, `pre×2` |
| `14/40-multi-session-handoff` | 27 | 32 | 84% | **C** (capped) | C1: 7q, `pre×2` |

**Tier pass rate (B+):** 0/4 = **0%**

---

### Tier E — Phase 19 capstone-projects (55 lessons, high-volume)

| Lesson | Raw | / max | % | Grade | P0 |
|--------|-----|-------|---|-------|-----|
| `19/22-jsonrpc-stdio-transport` | 25 | 36 | 69% | **C** (capped) | A2: off-lesson explanations |
| `19/24-plan-execute-control-flow` | 28 | 36 | 78% | **C** (capped) | A2: off-lesson explanations |
| `19/30-bpe-tokenizer-from-scratch` | 34 | 36 | 94% | **A** | none |
| `19/54-paper-writer` | 36 | 36 | 100% | **A** | none |

**Tier pass rate (B+):** 2/4 = **50%**

---

## Overall corpus pass rate

| Tier | Lessons sampled | B+ count | Pass rate |
|------|----------------|----------|-----------|
| A — Gold rebuilt | 4 | 4 | **100%** |
| B — Phase 04 | 4 | 0 | 0% |
| C — Phase 05 | 4 | 0 | 0% |
| D — Phase 14 | 4 | 0 | 0% |
| E — Phase 19 | 4 | 2 | 50% |
| **Total** | **20** | **6** | **30%** |

Target: ≥ 80% grade B+ per phase. **No phase reaches the target.** The newly rebuilt Phase 07/10 lessons are the only ones at 100%; every older phase has a structural schema issue affecting every lesson audited.

---

## P0 issues (ship-blockers)

| # | Issue | Affected scope | Fix |
|---|-------|---------------|-----|
| P0-1 | `07/15-attention-variants` check Q3: `correct` was **3** but explanation and doc both identify option **2** as correct | 1 lesson | **FIXED** in this session — changed `correct: 3 → 2` |
| P0-2 | **Phase 04 stage schema:** all quizzes have 5 questions with stages `pre, pre, post, post, post` instead of `pre, check, check, check, post, post` | All ~28 Phase 04 lessons | Add 3 `check` questions, drop 1 `pre`, drop 1 `post` per lesson; or regenerate via quiz-factory `redo_quiz` batch |
| P0-3 | **Phase 05 stage schema:** all quizzes have 7 questions with stages `pre×2, check×3, post×3` instead of 6 (`pre×1, check×3, post×2`) | All ~29 Phase 05 lessons | Drop 1 `pre` + 1 `post` per lesson, strengthen remaining post for integration |
| P0-4 | **Phase 14 stage schema:** all quizzes have 7 questions with `pre×2` | All ~42 Phase 14 lessons | Drop 1 `pre` per lesson; merge/rewrite as needed |
| P0-5 | **`19/22-jsonrpc-stdio-transport`** — all 6 explanations describe concepts not in `docs/en.md` (generic JSON-RPC / capstone metrics) | 1 lesson | Rewrite all explanations from `docs/en.md` sections |
| P0-6 | **`19/24-plan-execute-control-flow`** — all 6 explanations contain off-lesson content (task graphs, checkpointing, visualization) | 1 lesson | Rewrite all explanations from `docs/en.md` sections |

**Critical audit gap discovered:** `scripts/audit_lessons.py` Tier A does **not** check stage count or stage sequence. The P0-2 / P0-3 / P0-4 structural violations pass the current CI audit silently. Tier A must be extended to enforce the 6-question / stage-sequence requirement.

---

## P1 issues (fix before learner reaches the lesson)

| # | Issue | Scope | Criterion |
|---|-------|-------|-----------|
| P1-1 | `10/15-speculative-decoding-eagle3` check Q2: stem says acceptance improves "roughly doubles" for α=0.75→0.90; actual ratio is ~1.43×; option 0 cites ~5.7 tokens vs ~4.7 computed | 1 lesson | A2 / D2 |
| P1-2 | `07/15-attention-variants` check Q1: fabricated "1-bit quantization" distractor is implausible and untaught | 1 lesson | B7 |
| P1-3 | Phase 04 all lessons: no question references named code steps (`pad2d`, `UNet.forward`, `PatchEmbedding`, `rasterise_2d`) in any Build lesson | 4 lessons | A4 |
| P1-4 | Phase 05 all lessons: `B4` posts do not integrate two ideas (all are single-fact recall) | 4 lessons | B4 |
| P1-5 | Phase 14 all lessons: Build lessons do not reference `code/main.py` function names (`AgentLoop`, `generate_handoff`, workflow functions) | 4 lessons | A4 |
| P1-6 | `14/40-multi-session-handoff` pre Q0: answer is embedded in the option text ("status report, not a handoff") — guessable without reading | 1 lesson | B1 |
| P1-7 | `19/22-jsonrpc-stdio-transport`: no check/post references `StdioTransport`, `parse_request`, or `serve` | 1 lesson | A4 |
| P1-8 | `19/24-plan-execute-control-flow` post questions both single-fact recall (`revised` field, `replan_budget`) | 1 lesson | B4 |
| P1-9 | `19/30-bpe-tokenizer-from-scratch`: no check/post names `encode`, `decode`, or `allow_special` from `code/main.py` | 1 lesson | A4 |

---

## P2 issues (improve later)

| # | Issue | Scope |
|---|-------|-------|
| P2-1 | `05/27-llm-evaluation-frameworks`: several distractors use unrelated concepts ("GDPR", "tokenizer drift") not present in the doc | 1 lesson |
| P2-2 | `14/12-anthropic-workflow-patterns`: second pre Q embeds one check-level concept that could be merged | 1 lesson |
| P2-3 | `04/07-semantic-segmentation-unet`: one distractor ("optimizer failed") is too thin to be plausible | 1 lesson |
| P2-4 | `04/14-vision-transformers` pre Q1: "sorts pixel intensities" option is absurd | 1 lesson |
| P2-5 | `19/54-paper-writer` check 1 explanation mentions "unknown figure refs" not listed in doc validation gates | 1 lesson |

---

## Phase-level summary

| Phase | # lessons (total) | Schema issue | Structural fix needed | Content quality (raw %) |
|-------|------------------|--------------|-----------------------|------------------------|
| 04 computer-vision | ~28 | P0: 5q, wrong stages | All lessons | High (88–94%) |
| 05 NLP | ~29 | P0: 7q, extra pre+post | All lessons | High (84–97%) |
| 07 transformers | 17 | None found (rebuilt) | One wrong answer fixed | Excellent (97–100%) |
| 10 LLMs from scratch | 36 | None found (rebuilt) | None | Excellent (97–100%) |
| 14 agent-engineering | ~42 | P0: 7q, extra pre | All lessons | High (84–100%) |
| 19 capstone | ~55 | 2 P0 off-lesson explanations | 2 lessons | Mixed (69–100%) |

---

## Action queue

### Immediate (P0 — before any new phase ships)

1. **Fix `audit_lessons.py` Tier A** to enforce:
   - Exactly 6 questions per quiz
   - Stage sequence `pre, check, check, check, post, post`
   This unblocks automated detection for ~100 affected lessons across phases 04, 05, 14.

2. **Phase 04 schema repair batch** — queue all ~28 Phase 04 lessons as `redo_quiz` in `quiz-factory/manifest.json`. The structural pattern is uniform; a single batch run can repair them.

3. **Phase 05 schema repair batch** — same approach: drop 1 pre + 1 post per lesson across ~29 lessons, strengthen remaining post to integrate two doc ideas.

4. **Phase 14 schema repair batch** — drop 1 pre per lesson across ~42 lessons.

5. **Rewrite explanations for `19/22` and `19/24`** — both quizzes have technically valid stage schemas but off-lesson explanation content. These are targeted rewrites, not full rebuilds.

### Short-term (P1 — before the phase goes live)

6. Fix `10/15-speculative-decoding-eagle3` check Q2 math (acceptance rate overstated).
7. Fix `07/15-attention-variants` check Q1 distractor (remove "1-bit quantization" fiction).
8. For all Build lessons audited: add at least one check/post that names a specific function from `code/main.py`.

### Future (P2 — polish pass)

9. Distractor quality review for Phase 05 lessons with thin/absurd wrong options.
10. Phase 14: `14/40` pre Q0 rewrite (answer currently embedded in option text).

---

## Recommendations for the quiz-factory

1. **Update `audit_lessons.py` Tier A** with a stage-sequence / count check (one new rule `L003` or extend `L001`). This single change would have caught all P0-2 / P0-3 / P0-4 before merge.

2. **Template quizzes for phase repair batches** — a `BATCH.md` note for phases 04, 05, 14 should document the removal pattern so Claude Code knows to delete the extra `pre`/`post` and fill in the missing `check` questions from the doc.

3. **`19/54-paper-writer` and `19/30-bpe-tokenizer-from-scratch` are reference models** — use them as `style_anchor` entries in Phase 19's `BATCH.md` for the remaining 53 lessons.

4. **Gold standard quizzes from today's rebuild** (`07/16`, `10/34`) should be added as `style_anchor` examples in Phase 07 and Phase 10 `BATCH.md` files.

---

*Generated 2026-05-30. Nothing committed — findings require review before any quiz edits land.*

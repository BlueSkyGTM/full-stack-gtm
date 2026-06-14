# The Tailors — Accuracy & Cross-Reference Squad

Second squad in the GLM ecosystem. **The Loom weaves the cloth (generates lessons); the Tailors check the fit and finish the seams (verify accuracy, cross-reference sources).** Same ecosystem, different job, **separate manifest** — so the Tailors never compete with the Loom's generation flow.

The agent briefs were always titled "*Tailoring* Briefs" (`echo-brief.md`, `newton-brief.md`, `hypatia-brief.md`). This file names the squad and upgrades it.

---

## The upgrade (why this is new, not just a rename)

The original Echo and Newton were **GLM-5-Turbo, read-only, single-source** — Echo scanned the filesystem, Newton matched one citation at a time. Too shallow to do what's now needed: cross-reference each lesson's AI engineering against a **panel** of authoritative sources, not just the Made With ML extract. So the Tailors are promoted to **GLM-5.2** — the same 1M-context reasoning that made it a good Loom Taskmaster, now pointed at *accuracy* instead of generation. A Tailor holds the lesson + several primary sources in one window and reasons about whether the math, the code, and the framing are correct.

---

## The Squad (Echo + 4)

**Model:** all Tailors run `glm-5.2`. They **verify and flag** — they do not rewrite. A flagged lesson routes back to the Loom for a targeted correction. (Tailors finish seams; they don't re-weave the cloth.)

| Tailor | Domain | Cross-reference panel (beyond Made With ML) | Catches |
|--------|--------|---------------------------------------------|---------|
| **Echo** *(lead)* | Cross-source archaeology + squad orchestration | Made With ML · d2l.ai · Goodfellow *Deep Learning* · Karpathy zero-to-hero · the source curriculum | Drift from authoritative structure; routes each lesson to the right specialist |
| **Newton** | Mathematical & derivational correctness | Strang (linear algebra) · calculus texts · the primary derivations | Wrong gradients, bad √d_k scaling, broken chain-rule/backprop, malformed loss derivations |
| **Turing** | Algorithm & code correctness | nanoGPT · PyTorch docs · each paper's reference pseudocode | Build It that doesn't run or computes the wrong thing; false complexity/Big-O claims |
| **Shannon** | ML-theory & statistical correctness | Bishop *PRML* · MacKay *Information Theory* · primary stats sources | Wrong probability/entropy/KL, mis-stated cross-entropy, embedding-theory errors |
| **Hinton** | Modern deep-learning currency | recent arXiv · current framework docs (transformers, LLMs, agents) | Stale 2018-era framing; content that no longer matches current practice |

**Echo's expanded mandate (the Director's ask):** Echo no longer checks against a single Made With ML extract. She cross-references each lesson against the **panel** above, and assigns the lesson to the specialist whose domain owns the hardest claims (math → Newton, code → Turing, theory → Shannon, modern DL → Hinton).

**Hypatia** is the **GTM-strand tailor** — the sister role on the other helix strand (GTM source fidelity, double-helix alignment, coherence; `GLM-5.1`, binding PASS/WARN/BLOCK verdicts). The Tailors finish the AI seam; Hypatia finishes the GTM seam. Together they tailor both strands of the helix.

---

## Domain → Zone map (who audits what)

- **Newton** → 01 math-foundations
- **Shannon** → 02 ml-fundamentals, embeddings, probability/info-theory lessons
- **Hinton** → 03 deep-learning-core · 07 transformers · 08 sequence · 10 LLMs · 14 agents · 16 swarms
- **Turing** → every Build It block, all zones (code is cross-cutting)
- **Echo** → all zones (cross-source pass + routing)

---

## How the Tailors stay out of the Loom's flow

This is the design constraint. Three mechanisms:

1. **Separate manifest.** The Tailors run their own ICL loop on `stages/09-quality-pass/output/manifest.json` — they never touch the Loom's `stages/02-.../manifest.json`. Each squad reads and writes only its own map.
2. **Sequential chaining (ICL).** The Tailors' loop **opens only after** the Loom's generation + ship-ready pass closes (`pending = 0`). You verify finished content, not in-flight lessons. Loom builds → ship-ready → **Tailors verify** → Loom corrects flagged items. No two squads write the same lesson at the same time.
3. **Verify-flag, don't rewrite.** Tailors emit verdicts + the authoritative cross-reference; corrections route back to the Loom. This keeps the worker contention sequential, never concurrent. (Both squads share the Z.ai 5-call ceiling — sequential is what respects it.)

```
The Loom (build)  ──►  ship-ready pass  ──►  The Tailors (verify, flag)  ──►  Loom correction run
   GLM-5.2 + 5.1                              GLM-5.2 × 5, own manifest         targeted, flagged rows only
```

---

## What a Tailor produces

Per lesson, one verdict block to `stages/09-quality-pass/output/<zone>/<lesson>/audit.md`:

```
PASS  | 07-transformers/02-self-attention | Newton: √d_k scaling correct (Vaswani §3.2.1). Turing: Build It runs, shapes verified.
FLAG  | 10-llms/04-pre-training-mini-gpt  | Hinton: "GPT-2 is state of the art" is stale framing — reframe to historical. Newton: betas (0.9,0.95) correct per GPT-3.
BLOCK | 03-deep-learning/03-backprop       | Newton: gradient for layer 2 is transposed wrong vs the chain-rule derivation — Loom correction required.
```

`BLOCK` routes the row back to the Loom as a `failed` row with the gap reason. `FLAG` is a non-blocking correction ticket. `PASS` clears the AI seam.

---

## Running the Tailors (when Stage 09 opens)

Same launcher pattern as the Loom — a GLM-5.2 orchestrator over the Stage 09 manifest:

```powershell
export $(grep -v '^#' .env | xargs)
python skills/operator-kit/tailors.py --dry-run --sample 5   # confirm routing
python skills/operator-kit/tailors.py --tailor newton        # one specialist
python skills/operator-kit/tailors.py --workers 5            # full audit pass
```

`tailors.py` is built when Stage 09 runs (it mirrors `orchestrator.py`: read manifest → Echo routes → specialist GLM-5.2 verifies against its panel → write verdict → corrections back to the Loom). Not built yet — the Loom's generation + ship-ready must close first.

---

## Roster (full GLM ecosystem)

| Squad | Members | Model | Job | Manifest |
|-------|---------|-------|-----|----------|
| **The Loom** | GLM-5.2 Taskmaster + Lyra (GLM-5.1/5.1v) handlers | 5.2 + 5.1 | Weave lessons | `stages/02-.../manifest.json` |
| **The Tailors** | Echo · Newton · Turing · Shannon · Hinton | GLM-5.2 | Verify AI seam, cross-reference | `stages/09-.../manifest.json` |
| **GTM seam** | Hypatia | GLM-5.1 | GTM fidelity + coherence | shares Stage 09 |

The Conductor conducts both squads; neither runs while the other writes the same content.

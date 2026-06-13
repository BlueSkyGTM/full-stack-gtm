# Self-Refine and CRITIC: Iterative Output Improvement

---

## Beat 1: Hook

**Opening tension:** Your first LLM output is rarely your best. The generate-critique-refine loop is how you close that gap — without human intervention in the loop. Two papers formalize this: Self-Refine (internal feedback only) and CRITIC (external tool verification). The mechanism is the same: output → evaluate → revise → repeat until convergence or budget exhaustion.

---

## Beat 2: Concept

**Mechanism breakdown:**

- **Self-Refine** (Madaan et al., 2023): The model generates, then critiques its own output against specified criteria, then revises. No external ground truth. Relies on the model's ability to discriminate quality even when it can't produce quality in one shot.
- **CRITIC** (Gou et al., 2023): Same loop, but the critique step uses external tools — search, code execution, fact-checking APIs. The model verifies against evidence, not just self-judgment.
- **Stopping criteria:** Fixed iteration count, convergence detection (delta below threshold), or quality score exceeding a cutoff.
- **Failure mode:** Over-refinement. Models can sand away distinctive output into generic mush over too many iterations.

**Key distinction:** Self-Refine asks "does this meet the rubric?" CRITIC asks "is this factually grounded?" — different feedback signals, same loop structure.

---

## Beat 3: Demo

**Working code examples (terminal-only, observable output):**

1. **Minimal Self-Refine loop** — generate a product description, critique against a rubric (clarity, specificity, tone), revise, print all iterations so the practitioner can observe convergence or degradation.

2. **CRITIC variant** — generate a factual claim about a company, use a mock verification function (simulating a tool call) to flag errors, revise based on the external signal.

3. **Convergence detection** — track a simple quality metric across iterations, stop when delta drops below threshold or max iterations hit.

All code prints each iteration's output and the critique, so the loop behavior is visible.

---

## Beat 4: Use It

**GTM cluster redirect:** Personalization at scale — specifically, iterative refinement of outbound messaging.

**Application:** A self-refine loop can take a first-draft outreach email, critique it against ICP fit criteria (industry relevance, pain point specificity, tone match), and revise across 2–3 passes. This maps to the personalized outbound workflows in [CITATION NEEDED — concept: personalization refinement pipeline in GTM topic map].

**Application:** CRITIC-style verification can cross-check AI-generated account research against a knowledge base or search results before surfacing it to an SDR — preventing hallucinated firmographic claims. Maps to [CITATION NEEDED — concept: research verification in GTM topic map].

**Exercise hooks:**
- *(Easy)* Run the provided self-refine demo on 3 different prompts. Log which iteration produces the best output. Identify if the best is always the last.
- *(Medium)* Modify the rubric in the self-refine loop to optimize for a GTM criterion (e.g., "mentions a specific pain point for the target persona"). Compare outputs.
- *(Hard)* Implement a CRITIC loop that generates account summaries, verifies revenue/employee claims against a provided mock data source, and revises. Track error rate reduction per iteration.

---

## Beat 5: Ship It

**Integration exercise:** Build a refinement pipeline that takes a batch of inputs, runs configurable self-refine or CRITIC iterations, and writes the final outputs to a JSONL file with iteration count and final quality score.

**Extension:** Add a cost tracker — count total tokens across all iterations so the practitioner can evaluate whether 3 iterations of refinement are worth the 3x inference cost.

**Exercise hooks:**
- *(Easy)* Ship the pipeline with fixed 2-iteration self-refine on 5 inputs. Print per-input cost.
- *(Medium)* Add convergence-based early stopping. Compare total cost vs fixed-iteration cost on the same batch.
- *(Hard)* Implement both self-refine and CRITIC modes in the same pipeline. Run both on the same inputs. Write a comparison script that reports quality delta and cost delta.

---

## Beat 6: Troubleshoot

**Common failure modes:**

- **Sandwich effect:** Output improves on iteration 2, degrades by iteration 4 as the model "averages out" distinctive content. Observable in the demo — practitioner should see this happen.
- **Rubric gaming:** Model learns to satisfy the critique rubric in superficial ways (adding filler words that match tone criteria without improving substance).
- **CRITIC tool failures:** External verification source returns no data or conflicting data — model either ignores the signal or rewrites to hedging language that says nothing.
- **Cost blowup:** No iteration cap means a convergence failure turns into an infinite loop. Always set a hard max.

**Diagnostic pattern:** If iteration 3 is worse than iteration 1, the rubric is wrong or the task doesn't benefit from refinement. Not everything should be refined — classification tasks, structured extraction, and format conversion often degrade with extra passes.

---

## Learning Objectives

1. **Implement** a self-refine loop with configurable iteration count and observable intermediate outputs.
2. **Compare** self-refine (internal critique) vs CRITIC (external tool verification) feedback mechanisms and their tradeoffs.
3. **Evaluate** output quality across iterations to detect over-refinement and diminishing returns.
4. **Configure** convergence-based stopping criteria using measurable quality signals.
5. **Build** a batch refinement pipeline that tracks both quality improvement and token cost per iteration.
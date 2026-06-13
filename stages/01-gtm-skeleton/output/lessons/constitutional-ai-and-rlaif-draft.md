# Constitutional AI and RLAIF

## Learning Objectives

1. Compare RLAIF against RLHF in terms of feedback source, cost, and scalability constraints.
2. Implement a constitutional self-critique loop that generates, evaluates, and revises outputs against an explicit principle set.
3. Construct a preference dataset from AI-generated critiques and revisions.
4. Evaluate the effect of constitutional principles on output alignment using measurable criteria.
5. Diagnose failure modes where AI-generated feedback inherits model biases rather than correcting them.

---

## Beat 1: Hook

Why AI feedback loops matter when you can't scale human labelers — and why "just use RLHF" stops working at production volume. The cost and latency wall of human preference data. The credibility problem: if the judge and the defendant are the same model, what actually gets enforced?

---

## Beat 2: Concept

**Mechanism breakdown:**

- The two-phase pipeline: supervised self-critique (SL) followed by reinforcement learning from AI preferences (RL).
- Phase 1 (SL): model generates response → model critiques against constitutional principles → model revises → fine-tune on revisions.
- Phase 2 (RL): model generates pair of responses → AI evaluates which is better per constitution → preference model trained on AI labels → RL (PPO or DPO) optimizes against preference model.
- RLAIF vs RLHF: same training loop, different label source. AI replaces human annotators in the preference judgment step.
- The constitution itself: an ordered list of natural-language rules. Order matters — earlier principles dominate in conflicts. [CITATION NEEDED — concept: principle ordering priority in constitutional evaluation]
- Failure mode: "reward hacking" where the model learns to satisfy the letter of the constitution while violating its intent. Also "sycophancy" — the AI judge may rate outputs that agree with the prompt author more highly.

**Key distinction:** RLAIF is the general technique. Constitutional AI is Anthropic's specific instantiation with an explicit principle list.

---

## Beat 3: Demo

**Demo 1: Minimal self-critique loop**
- Hardcode a 3-item constitution (e.g., "Be concise", "Do not fabricate facts", "Acknowledge uncertainty").
- Generate an initial response to a prompt.
- Feed response + constitution back to the model for critique.
- Generate revision.
- Print: original, critique, revision — with side-by-side comparison.

**Demo 2: AI preference judgment**
- Generate two responses to the same prompt.
- Ask the model to evaluate both against constitutional principles and pick one.
- Output: both responses, the reasoning, and the selection — demonstrating the preference signal that would feed a reward model.

Observable output: printed transcripts showing the critique and revision cycle, and preference selection with reasoning.

---

## Beat 4: Use It

**GTM Redirect: Zone 04 — AI Safety and Alignment in Production Systems**

[CITATION NEEDED — concept: GTM cluster for AI-assisted outbound quality assurance]

The self-critique loop is a direct template for outbound message QA. In GTM contexts, your "constitution" becomes your brand guidelines, compliance rules, and tone standards. Instead of human review of every sequence step, a constitutional evaluator checks generated outreach against written principles.

The preference signal mechanism maps to A/B test evaluation: two subject lines, two value props — an AI judge scores both against your brand constitution, producing a ranking without human review time. The same failure modes apply: if your brand guidelines are vague, the model will game them.

---

## Beat 5: Ship It

**Exercise hooks:**

- **Easy:** Write a 5-item constitution for a specific domain (e.g., technical documentation, sales outreach). Run the self-critique loop on 3 prompts and report which principles were violated most frequently.
- **Medium:** Implement a preference dataset generator: produce 10 prompt-response pairs, run AI evaluation, and output a JSONL dataset formatted for DPO training with chosen/rejected fields.
- **Hard:** Diagnose sycophancy. Design 5 adversarial prompts where the prompt author implies a preferred answer. Run the constitutional judge with and without the author's implied preference visible in context. Measure whether the constitution overrides sycophancy — report the delta.

---

## Beat 6: Digest

**Key takeaways:**

- RLAIF replaces human labelers with AI judges; Constitutional AI is one specific implementation.
- The self-critique pipeline has two distinct phases: supervised fine-tuning on revisions, then RL from AI preferences.
- A constitution is ordered natural-language rules — earlier rules win conflicts.
- The central failure mode is self-dealing: the model evaluating itself inherits its own blind spots.
- For GTM applications, your "constitution" is your brand/compliance document — the technique is the same, the principles change.

**Further reading:**
- Bai et al. (2022), "Constitutional AI: Harmlessness from AI Feedback"
- Lee et al. (2023), "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback"
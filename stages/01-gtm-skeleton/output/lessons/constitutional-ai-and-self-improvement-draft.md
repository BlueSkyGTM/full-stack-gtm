# Constitutional AI and Self-Improvement

## Learning Objectives

1. Implement a generate-critique-revise loop that evaluates AI outputs against a defined set of principles
2. Write constitutional principles that produce measurable, testable quality gates for AI-generated content
3. Detect failure modes in self-critique loops, including sycophancy and principle conflicts
4. Compare constitutional self-evaluation against held-out human labels to measure alignment accuracy
5. Configure a multi-principle constitution for a GTM content pipeline and evaluate revision quality

---

## Beat 1: Hook

The problem: you cannot human-label every output an AI system produces in production. You need a scalable mechanism to keep behavior within bounds. Constitutional AI proposes one answer — the model critiques itself against written principles, then revises. The question is whether that loop actually converges or just circles.

---

## Beat 2: Concept

Constitutional AI (Anthropic, 2022) replaces pure human preference labeling with a two-phase process: (1) the model generates a response, critiques it against a written constitution, and revises; (2) the revised responses become training data for preference learning. The "constitution" is an ordered list of natural-language rules. The key claim: self-critique against principles can substitute for large-scale human feedback. The key risk: the model can learn to endorse its own outputs without meaningfully changing them — sycophancy with extra steps.

**Key terms to define:**
- Constitution: ordered list of natural-language principles
- Self-critique: model evaluates its own output against a principle
- Revision: model rewrites output given critique
- RL from AI Feedback (RLAIF): preference training on AI-generated judgments instead of human labels
- Sycophancy: model agrees with its own prior output regardless of principle

---

## Beat 3: Mechanism

### Phase 1: Supervised Self-Critique (SL)

The algorithm:
1. Prompt the model with a task
2. Generate an initial response
3. Append a critique prompt: "Identify specific ways this response violates the following principle: <principle>"
4. Generate critique
5. Append a revision prompt: "Rewrite the response to address the critique above"
6. Generate revised response
7. The (prompt, revised_response) pair becomes supervised training data

### Phase 2: Preference Learning (RL from AI Feedback)

1. For each prompt, generate two responses
2. Ask the model to evaluate which response better satisfies the constitution
3. Use these pairwise preferences to train a preference model
4. Use the preference model as reward signal for RL fine-tuning

### Failure modes:
- **Tautological critique**: "This response is good because it follows the principle" — no actionable revision
- **Principle collision**: Principle A and Principle B contradict; model oscillates
- **Reward hacking**: Model learns critique patterns that score well but don't improve output quality
- **Distribution collapse**: Over-revision makes all outputs sound identical

**Code example**: Build a single-pass generate-critique-revise loop with observable output. Show the original response, the critique, and the revised response. Use a local principle set. Print all three stages.

```
[Exercise hooks:
- Easy: Run the provided generate-critique-revise loop on three different prompts and log where the revision improves vs. doesn't
- Medium: Add a second principle and detect cases where principles conflict in the critique
- Hard: Implement a pairwise preference evaluator and measure agreement rate with human labels on 20 examples]
```

---

## Beat 4: Use It

### GTM Redirect: Automated Quality Gates for AI-Generated Outreach

**Cluster reference**: [CITATION NEEDED — concept: constitutional AI mapping to GTM content quality pipeline]

The constitutional pattern maps to any GTM workflow where AI generates customer-facing content at scale. The "constitution" becomes your brand guidelines, compliance rules, and tone standards. The generate-critique-revise loop becomes a pre-send quality gate.

Specific application: personalization pipelines. You generate personalized outreach at scale. You cannot review every message manually. A constitutional loop evaluates each message against your principles before it enters the send queue.

**What this is not**: This is not a replacement for human review of high-stakes outputs. This is a scalable filter that catches the bottom 30% of quality issues automatically.

**Code example**: Build a constitutional evaluator for outreach messages. Define 3-4 principles (tone, factual accuracy check, length constraint, nospam markers). Run a batch of generated messages through critique-revise. Print before/after with pass/fail flags.

```
[Exercise hooks:
- Easy: Define 3 outreach principles and run 10 messages through the critique filter; report pass rate
- Medium: Add a "brand voice" principle and measure how often revision changes the CTA structure
- Hard: Build a scorer that compares pre-revision and post-revision messages against human-rated examples; report precision/recall]
```

---

## Beat 5: Ship It

### Production Constitutional Loop

Ship a self-critique pipeline that processes outputs in batch. Architecture:

1. **Principle store**: JSON or YAML file containing ordered principles with severity levels
2. **Generator**: Produces initial outputs
3. **Critic**: Evaluates each output against each principle; returns structured critique (violation found: yes/no, severity, specific text)
4. **Reviser**: Rewrites only if violations detected
5. **Logger**: Records original, critique, revision, and principle violations for monitoring
6. **Escalation gate**: If revision still fails principles after N attempts, flag for human review

Key production concerns:
- Latency: each critique-revise cycle adds API calls; budget N cycles max
- Cost: self-critique triples token usage per output
- Monitoring: track violation rates by principle over time — if a principle triggers on >50% of outputs, it may be too strict or the generator needs retraining
- Convergence detection: if revision doesn't change the output meaningfully, stop cycling

**Code example**: Build the full pipeline with a principle store (YAML/JSON), batch processing, logging to file, and an escalation gate. Process a batch of 20 messages. Print summary statistics (pass rate, violation distribution, revision delta, escalations).

```
[Exercise hooks:
- Easy: Run the pipeline on the provided message batch; produce the summary report
- Medium: Modify the principle store to include a domain-specific rule; measure the change in violation patterns
- Hard: Implement convergence detection (compare revision N to revision N-1 using string similarity); report how many messages converge vs. oscillate]
```

---

## Beat 6: Evaluate

### Does It Actually Work?

Measuring constitutional AI effectiveness requires external ground truth — you cannot evaluate self-critique with self-critique.

**Evaluation protocol:**
1. **Holdout human labels**: Label 50-100 outputs as acceptable/unacceptable. Measure whether the constitutional loop's pass/fail decisions agree with human judgment (precision, recall, F1).
2. **Revision quality**: For outputs that get revised, do humans prefer the revision over the original? Run a blinded A/B test.
3. **Convergence audit**: Sample revised outputs and check if the critique identified real problems or produced tautological observations.
4. **Failure mode log**: Track and categorize failure modes — sycophantic critiques, principle collisions, distribution collapse.
5. **Cost-benefit**: Calculate the cost of running the constitutional loop (tokens, latency) against the cost of human review it replaces.

**Anti-patterns to watch:**
- Pass rate >95%: the constitution is too weak or the model is rubber-stamping
- Revision rarely changes output: critique is superficial
- High variance in violation detection across runs: the model is not deterministic enough for reliable quality gates

**Code example**: Build an evaluation script that compares constitutional loop outputs against a provided set of human labels. Calculate precision, recall, F1. Print a confusion matrix and a per-principle breakdown.

```
[Exercise hooks:
- Easy: Run the evaluation script on the provided human labels; report overall F1
- Medium: Identify which principle has the lowest precision; hypothesize why and test a revision to the principle wording
- Hard: Run the full pipeline on 50 outputs, have a human label 20, and measure inter-rater agreement between the constitutional loop and the human]
```

---

## docs/en.md Appendix

```
# Constitutional AI and Self-Improvement

## Prerequisites
- Lesson: Prompt Engineering Foundations
- Lesson: Evaluation Methods for LLM Outputs

## Objectives
1. Implement a generate-critique-revise loop that evaluates AI outputs against a defined set of principles
2. Write constitutional principles that produce measurable, testable quality gates for AI-generated content
3. Detect failure modes in self-critique loops, including sycophancy and principle conflicts
4. Compare constitutional self-evaluation against held-out human labels to measure alignment accuracy
5. Configure a multi-principle constitution for a GTM content pipeline and evaluate revision quality

## Key Terms
- Constitution: ordered list of natural-language evaluation principles
- Self-critique: model evaluates its own output against principles
- RLAIF: Reinforcement Learning from AI Feedback (vs. RLHF)
- Sycophancy: model agrees with itself without meaningful critique
- Convergence: revised output stabilizes across critique cycles

## Failure Modes
- Tautological critique
- Principle collision
- Reward hacking
- Distribution collapse
- Rubber-stamping (pass rate >95%)

## GTM Application
- Scalable quality gates for AI-generated outreach and content
- Pre-send constitutional evaluation in personalization pipelines
- Not a replacement for human review of high-stakes outputs
```
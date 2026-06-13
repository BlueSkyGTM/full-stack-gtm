# Few-Shot, Chain-of-Thought, Tree-of-Thought Prompting

## Learning Objectives
1. Compare few-shot, chain-of-thought, and tree-of-thought prompting by their mechanism of action and tradeoffs.
2. Implement few-shot classification with a fixed example budget and verify output consistency.
3. Construct chain-of-thought prompts that produce visible intermediate reasoning steps.
4. Evaluate whether tree-of-thought exploration yields measurably better outputs than linear chain-of-thought for a given task.
5. Diagnose which prompting strategy fits a task based on task properties (classification vs. multi-step reasoning vs. open-ended planning).

---

## Beat 1: Hook

The through-line: these three techniques form an escalation ladder. Few-shot teaches the model *what* by example. Chain-of-thought teaches it *how* by forcing visible reasoning. Tree-of-thought teaches it *where* by exploring multiple paths. The practitioner's job is picking the right rung — because each step up costs more tokens and latency.

---

## Beat 2: Explain

**Few-shot mechanism:** Provide K input→output pairs in the prompt. The model performs in-context learning — it pattern-matches against the examples at inference time without weight updates. The decision: how many shots, what distribution of examples, and what to do when the model drifts off-pattern.

**Chain-of-thought (CoT) mechanism:** Append reasoning traces to the examples (or just prompt "think step by step"). The model generates intermediate tokens that act as scratchpad computation. Research suggests this works because transformers can only perform so much computation per token — spreading computation across more tokens increases effective depth [CITATION NEEDED — concept: CoT inference-time compute scaling, Wei et al. 2022].

**Tree-of-thought (ToT) mechanism:** Extend CoT by branching. Generate multiple candidate reasoning paths at each step, evaluate them (via self-critique or scoring), prune weak branches, and expand strong ones. This is beam search over language model reasoning. The implementation requires an orchestrator that calls the model multiple times per step — it is not a single-prompt technique [CITATION NEEDED — concept: ToT as search over language, Yao et al. 2023].

Key distinction: few-shot is single-pass. CoT is single-pass with more tokens. ToT is multi-pass with orchestration logic.

---

## Beat 3: Show

**Few-shot demo:** A Python script that classifies support tickets into categories using 0-shot, 1-shot, and 3-shot prompts. Prints classification accuracy and consistency across runs to show the empirical impact of example count.

**CoT demo:** Same classification task, but the prompt includes reasoning traces ("This mentions 'billing' and 'charge', so the category is..."). Prints both the final answer and the intermediate reasoning. Compare token usage and correctness against the few-shot baseline.

**ToT demo:** A planning task (e.g., "write an outbound sequence for a CISO at a Series B fintech") where the script generates 3 initial approaches, scores each, expands the top 2, and returns the best plan. Prints the full tree: branches explored, scores, and final selection. Uses only `anthropic` SDK calls and a scoring function — no framework dependency.

Each demo is a standalone script that runs in terminal and prints observable output.

---

## Beat 4: Use It

**GTM redirect:** This maps to the **Enrichment → Personalization** cluster and the **Outbound Messaging** cluster in `stages/00-b-gtm-content-mapping/output/gtm-topic-map.md`.

- **Few-shot** in GTM: You have 5 examples of good personalized icebreakers. You feed them to the model with a new prospect's LinkedIn data. The model pattern-matches against your examples. This is how you maintain voice consistency across an SDR team's AI-assisted outreach.
- **Chain-of-thought** in GTM: Account scoring that requires multi-step inference — "This company raised Series B (signal), they use AWS (intent), their CISO spoke at a security conference (intent), therefore they likely need X." CoT forces the model to surface each step so a human can audit the scoring logic.
- **Tree-of-thought** in GTM: Campaign strategy generation where you want to explore multiple angles (pain-point-led, trigger-led, social-proof-led) before committing. ToT lets you generate and evaluate branches.

Exercise hooks:
- **Easy:** Write a 3-shot prompt that classifies inbound leads by ICP fit. Run it on 10 test leads and print results.
- **Medium:** Build a CoT-based account scorer that outputs its reasoning. Verify the reasoning is actually used by running the same input with and without CoT and comparing outputs.
- **Hard:** Implement a ToT orchestrator for outreach angle generation. Generate 3 angles, self-critique each, expand the winner. Print full tree with scores.

---

## Beat 5: Ship It

**GTM redirect:** Same clusters — this is the production version of Use It.

Build a CLI tool that accepts a prospect's LinkedIn profile JSON and a company news item, then:
1. Uses few-shot to classify the persona (ICP tier).
2. Uses CoT to generate a scoring rationale.
3. Optionally uses ToT to explore multiple outreach angles when the score is borderline.

The tool prints structured output: tier, score, rationale, and suggested first line. No Clay dependency — this is raw API calls with the prompting patterns applied.

Exercise hooks:
- **Easy:** Ship step 1 only — few-shot persona classifier with test inputs.
- **Medium:** Ship steps 1 + 2 — classifier plus CoT scorer with auditable reasoning.
- **Hard:** Ship all three with ToT branching on borderline cases. Add a flag to toggle ToT on/off and compare output quality.

---

## Beat 6: Assess

Questions that test mechanism-level understanding, not trivia:

1. Given a task description, select the appropriate prompting strategy and justify why the others are worse. (Tests objective 5.)
2. A few-shot prompt produces inconsistent outputs across runs with the same input. Diagnose: is this an example problem, a temperature problem, or a task complexity problem? (Tests objective 2.)
3. A CoT prompt produces correct reasoning but the wrong final answer. Identify the failure mode. (Tests objective 3.)
4. Compare token costs and latency for CoT vs. ToT on the same task. Calculate the tradeoff. (Tests objective 4.)
5. Implement a new few-shot prompt from scratch for a novel GTM classification task not covered in the lesson. (Tests objective 2.)

Each question maps directly to a learning objective. No "which of the following is true about chain-of-thought" style trivia.
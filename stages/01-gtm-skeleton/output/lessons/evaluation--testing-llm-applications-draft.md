# Evaluation & Testing LLM Applications

---

## Hook

You shipped a prompt change and classification accuracy on support tickets quietly dropped 12%. No alert fired because no eval existed. This beat makes the case that evals are to LLM apps what tests are to traditional software — except harder, because the outputs are unstructured text with no single correct answer.

---

## Concept

Define the three layers of LLM evaluation: **static test sets** (golden input-output pairs), **automated judges** (heuristic + LLM-as-judge scoring), and **runtime monitoring** (drift detection in production). Explain the mechanism of each: test sets compare against reference outputs using exact match / semantic similarity; LLM-as-judge uses a second model with a rubric; runtime monitoring tracks distribution shifts. Introduce the vocabulary: *candidate*, *reference*, *metric*, *rubric*, *regression*. Acknowledge the hard problem — "correct" is subjective for generative tasks, which is why rubric design is the actual engineering work.

---

## Demo

Build a minimal eval harness from scratch: load a JSONL test set of 20 labeled examples (e.g., intent classification or reply categorization), run each through an LLM call, score with three metrics (exact match, substring match, LLM-as-judge with a simple rubric), and print a summary table showing per-example pass/fail plus aggregate accuracy. All code runs in terminal, prints observable output. No framework dependency — this is the mechanism before naming tools like Promptfoo, Braintrust, or LangSmith.

---

## Use It

**GTM Redirect → Cluster 11: Living GTM — Revenue Intelligence.**

Evals are how you A/B test your outbound sequences before they go live. If you're running reply classification (positive interest / objection / unsubscribe), your eval set is your labeled reply corpus, and your metric is classification F1 against human-labeled examples. Every time you change a prompt, you rerun the eval — this is the feedback loop that turns reply data into a measurable system, not a vibe check. [CITATION NEEDED — concept: eval-driven sequence optimization in outbound workflows]

**Exercise hooks:**
- *Easy*: Label 10 replies manually, build a JSONL test set, run exact-match eval against a fixed classifier prompt.
- *Medium*: Add an LLM-as-judge scorer with a rubric that grades classification confidence, compare scores to exact match.
- *Hard*: Simulate a prompt regression — modify your classifier prompt to be worse, prove the eval catches the regression with a diff report.

---

## Ship It

Cover production eval pipelines: where the test set lives (version-controlled alongside prompts), when evals run (pre-merge in CI, not post-deploy), and how to handle the "flaky test" problem when LLM outputs are non-deterministic (set temperature=0 for eval runs, accept tolerance bands, flag edge cases for human review). Name the tool landscape: Promptfoo implements config-driven eval matrices; Braintrust provides a hosted eval runner with scoring functions; both solve the "run-many-examples-against-many-prompts" problem. End with a decision rule: if the task has discrete outputs (classification, extraction), use exact/metric scoring; if the task is generative (drafts, summaries), use LLM-as-judge with an explicit rubric.

**Exercise hooks:**
- *Easy*: Write a shell script that runs your eval harness and exits non-zero if accuracy drops below a threshold.
- *Medium*: Wire the eval script into a git pre-commit hook so prompt changes are auto-tested before commit.
- *Hard*: Build a two-prompt comparison matrix (current vs. candidate prompt) that outputs a side-by-side score diff table for 30 examples.

---

## Quiz

Five questions grounded in the mechanisms above: identify which eval approach fits a given task type, read a scored output table and identify a regression, distinguish between test-set eval and runtime monitoring, select the right metric for a classification vs. generative task, debug a failing eval by diagnosing whether the rubric or the prompt is the problem. All questions trace to specific learning objectives.

---

**Learning Objectives (for `docs/en.md`):**

1. Build a labeled test set and eval harness for a classification or generation task.
2. Implement three scoring methods — exact match, heuristic similarity, LLM-as-judge — and compare their trade-offs.
3. Configure a regression test that catches accuracy degradation between prompt versions.
4. Evaluate when to use static test sets versus runtime drift monitoring in a production pipeline.
5. Diagnose whether a failing eval stems from the prompt, the rubric, or the test set itself.
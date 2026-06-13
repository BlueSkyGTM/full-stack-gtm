# Lesson: Language Model Evaluation Harness

---

## Beat 1: Hook

Every model vendor publishes benchmark scores. Most are not reproducible. The `lm-evaluation-harness` from EleutherAI is the open standard that makes those claims falsifiable — you run the same task suite, same prompt formatting, same metric computation, and compare. If you're selecting a model for any production pipeline, you need to know what this tool does and what its output actually means.

---

## Beat 2: Concept

The mechanism: a task is a YAML config + dataset that defines few-shot examples, prompt template, and metric (accuracy, F1, perplexity). The harness loads a model, formats prompts according to task config, collects logits or generations, computes metrics, and aggregates results. Key patterns: task registration, multiple-choice normalization, per-generation vs per-token scoring. Cover the difference between `loglikelihood`, `multiple_choice`, and `generate_until` task types. Then name the tool: `lm-eval` (EleutherAI).

---

## Beat 3: Demonstration

Install `lm-eval`, run a small model (e.g., `gpt2` or `hf-causal-experimental` pointing to a small local model) against 2–3 lightweight tasks (`hellaswag`, `arc_easy`, `mmlu` subset). Print results to terminal. Show what each field in the output dict means — `acc`, `acc_norm`, `stderr`. Demonstrate how to pass `--limit` for fast iteration during development.

---

## Beat 4: Use It

GTM redirect: model selection for GTM automation pipelines. [CITATION NEEDED — concept: GTM cluster for model selection/benchmarking in pipeline design]. When evaluating which LLM to use for lead enrichment, classification, or personalization tasks, benchmark data from `lm-eval` provides a falsifiable baseline beyond vendor claims. The harness does not tell you which model is "best for GTM" — it tells you which model scores what on which task. You map the task to your use case.

---

## Beat 5: Ship It

Exercise hooks:

- **Easy**: Run `lm-eval` with `--tasks arc_easy --limit 50` on `gpt2`, capture the JSON output, and print `acc` and `acc_norm` with their stderr values.
- **Medium**: Create a custom task YAML that evaluates a model on a classification dataset relevant to your domain (e.g., a CSV of support tickets with categories). Register it in `tasks/` and run it.
- **Hard**: Compare two models (e.g., `gpt2` vs `gpt2-medium`) across the same 5 tasks. Write a script that parses both result JSONs, computes the delta per task, and flags any result where stderr overlaps — meaning the difference is not statistically meaningful.

---

## Beat 6: Objectives

1. **Configure** and run `lm-eval` against a specified model and task list from the command line.
2. **Interpret** output metrics (`acc`, `acc_norm`, `perplexity`, `stderr`) and identify when a score difference is within noise.
3. **Create** a custom task YAML definition pointing to an arbitrary dataset and specifying prompt format and metric type.
4. **Compare** benchmark results across two or more models and produce a structured delta report.
5. **Explain** the difference between `loglikelihood`, `multiple_choice`, and `generate_until` task types and when each is appropriate.

---

## GTM Redirect Rules (lesson-level)

- **"Use It" redirect**: Model selection for GTM pipelines — benchmark data from `lm-eval` grounds model choice in reproducible metrics rather than vendor marketing.
- **"Ship It" redirect**: Custom task evaluation maps directly to testing whether a model handles your specific GTM classification or extraction task before deploying it in production.
- **Forced connection guard**: If no clean GTM mapping exists for a given evaluation task (e.g., pure reasoning benchmarks), the redirect is "foundational for AI engineering literacy" — not a fabricated pipeline application.
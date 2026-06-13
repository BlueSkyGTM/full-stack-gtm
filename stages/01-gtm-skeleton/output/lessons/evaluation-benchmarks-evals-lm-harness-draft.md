# Evaluation: Benchmarks, Evals, LM Harness

## Beat 1: Hook

You built the pipeline. It runs. But when you prompt it with "classify this lead: hot or not" and it returns a 500-word essay about lead scoring philosophy, you realize: deployment without evaluation is just hopeful debugging in production. This lesson covers the three layers of "does this actually work"—benchmarks, task-specific evals, and the harnesses that run them at scale.

## Beat 2: Concept

The evaluation stack has three distinct layers. **Benchmarks** (MMLU, HumanEval, GSM8K) measure a model's general capability against a fixed reference—these tell you which model to pick, not whether your specific prompt works. **Evals** are task-specific tests you write for your own use case: "given these 50 emails, does my classifier get ≥90% right?" **Harnesses** are the infrastructure that runs evals reproducibly, handles few-shot framing, and aggregates metrics. Conflating these three is the most common evaluation mistake in GTM AI work.

Key distinctions to nail down:
- Benchmark ≠ your eval. A model that aces MMLU can still fail at your specific lead-routing task.
- Eval without a harness = manual spot-checking. Harness without a custom eval = benchmark theater.
- The metric you choose (exact match, F1, BLEU, LLM-as-judge) determines what "good" means. Choose wrong and you optimize for the wrong behavior.

## Beat 3: Mechanism

**How benchmarks work:** A standardized dataset (question, correct answer pairs) is fed through the model with a fixed prompt template. The model's output is compared against references using a metric (exact match, multiple-choice probability). Scores are aggregated per-category and overall. MMLU has 57 subjects, ~14K questions. HumanEval has 164 Python programming problems with unit tests. The mechanism is simple; the interpretation is where people go wrong—a 2-point MMLU difference is noise, not signal.

**How custom evals work:** You define: (1) a dataset of inputs and expected outputs, (2) a prompt template that wraps inputs, (3) a scoring function that compares model output to expected output. The scoring function is where the engineering lives—exact match is brittle for generative tasks, semantic similarity catches meaning but lets hallucinations slide, LLM-as-judge introduces a second model's biases. For classification tasks (reply categorization, intent detection), F1 per class is standard. For extraction tasks (pulling company names, deal values), token-level F1 or Jaccard similarity works.

**How LM Evaluation Harness works:** The `lm-eval-harness` library (EleutherAI) implements this pattern: load a model via a unified interface, load a task config (benchmark or custom), generate outputs with configurable few-shot settings, score against references, report aggregated metrics. It handles the tedious parts: tokenization for likelihood-based tasks, batching, context window management, reproducibility via seeding.

The critical mechanism to understand: **few-shot contamination**. If your eval data leaked into the training set, your benchmark score is memorization, not capability. This is detectable but often ignored.

## Beat 4: Code

Build and run a custom eval harness from scratch. The code will:
1. Load a small model (or mock the interface for speed)
2. Define a custom task: classifying sales reply intent (interested / not interested / opt-out / question)
3. Run the eval with exact-match and F1 scoring
4. Print per-class precision, recall, F1, and a confusion matrix
5. Show how swapping the prompt template changes scores (the actual "prompt engineering" that matters)

Observable output: a printed eval report showing metrics and a before/after comparison when the prompt is changed.

## Beat 5: Use It

**GTM Redirect → Zone 11: Revenue Intelligence / Living GTM**

Evals are A/B testing your sequences before they go live. The mechanism is identical: define a task (reply classification), label a gold-standard dataset (50 real replies manually tagged), run your classifier against it, measure accuracy per class. The eval is your feedback loop—if "opt-out" detection has 60% recall, you're leaving unsubscribe requests in the queue.

Specific GTM evals to build:
- **Reply classification eval:** Precision/recall per intent category. The gold dataset is 100+ manually labeled replies from your actual inbox.
- **Personalization eval:** Given a lead profile + generated email, does the email contain factually correct claims about the lead's company? LLM-as-judge with a rubric, scored 1-5.
- **Extractive eval:** Given a call transcript, does the model extract the correct next steps, budget numbers, and timeline? Token-level F1 against human-annotated references.

The eval is the artifact that makes GTM AI trustworthy. Without it, you're deploying vibes.

## Beat 6: Ship It

**Easy:** Write a 20-item gold-standard dataset for reply classification (5 examples per class). Run exact-match eval against a zero-shot prompt. Print the confusion matrix.

**Medium:** Build a reusable eval harness class that accepts any (dataset, prompt_template, scoring_fn) triple. Run it with two different prompt templates on the same dataset and print a side-by-side comparison table.

**Hard:** Implement LLM-as-judge scoring for a personalization eval. The judge model receives (lead_profile, generated_email, rubric) and outputs a 1-5 score with justification. Run this on 10 generated emails, compute correlation with human ratings, and print the agreement matrix. Report where the judge disagrees with humans and hypothesize why.

---

**Learning Objectives:**
1. Compare benchmarks, custom evals, and eval harnesses—identify when each is appropriate.
2. Build a custom evaluation harness that runs a classification task and reports per-class precision, recall, and F1.
3. Configure a prompt template, run an eval, measure the score delta when the template changes.
4. Evaluate a generative task using LLM-as-judge scoring with a defined rubric.
5. Diagnose eval contamination and explain why benchmark scores do not predict task-specific performance.
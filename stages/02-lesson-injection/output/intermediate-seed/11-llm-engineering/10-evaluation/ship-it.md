## Ship It

Production eval pipelines need three things: where the test set lives, when evals run, and how to handle non-determinism.

The test set lives in version control alongside your prompts. Not in a database, not in a spreadsheet — in the same repository as the code that runs the prompts. When someone opens a pull request that changes a prompt, the diff includes both the prompt change and the eval results. This makes the trade-off visible: "I am changing the prompt to handle edge case X, and here is the proof that it does not break cases A through W."

Evals run pre-merge in CI, not post-deploy. The CI job loads the test set, runs the classifier against every example at `temperature=0`, scores the outputs, and compares aggregate accuracy to the baseline. If accuracy drops more than the tolerance band (say, 2 percentage points), the CI check fails and the merge is blocked. This is the same pattern as running unit tests in CI for traditional software — the mechanism is identical, the assertion logic is different.

Non-determinism is the hard part. Even at `temperature=0`, LLM outputs are not perfectly deterministic across model versions or API updates. Three strategies handle this. First, pin `temperature=0` for eval runs — this eliminates most variance. Second, set tolerance bands rather than hard thresholds — require accuracy >= 82% rather than accuracy == 84.3%, because a 0.5% fluctuation is noise, not signal. Third, flag edge cases for human review — when the judge disagrees with exact match, log it and review weekly. These disagreements are where your rubric needs refinement.

```python
import subprocess
import sys

def ci_eval_check(baseline_accuracy: float, candidate_accuracy: float, tolerance: float = 0.02) -> int:
    diff = baseline_accuracy - candidate_accuracy
    print(f"Baseline accuracy: {baseline_accuracy*100:.1f}%")
    print(f"Candidate accuracy: {candidate_accuracy*100:.1f}%")
    print(f"Delta: {diff*100:+.1f}%")
    print(f"Tolerance: ±{tolerance*100:.1f}%")

    if diff > tolerance:
        print(f"\nFAIL: Regression exceeds tolerance ({diff*100:.1f}% > {tolerance*100:.1f}%)")
        return 1
    else:
        print(f"\nPASS: Within tolerance")
        return 0

baseline_passes = sum(1 for r in baseline if r["exact_pass"])
candidate_passes = sum(1 for r in candidate if r["exact_pass"])
baseline_acc = baseline_passes / len(baseline)
candidate_acc = candidate_passes / len(candidate)

exit_code = ci_eval_check(baseline_acc, candidate_acc, tolerance=0.02)
sys.exit(exit_code)
```

The tool landscape: Promptfoo implements config-driven eval matrices where you define prompts, test cases, and assertions in YAML, and it runs the cross-product and produces a comparison table. Braintrust provides a hosted eval runner with experiment tracking — each eval run is logged with its prompt version, model, and scores, so you can compare runs over time. LangSmith offers tracing and eval integration tied to LLM call logs. The mechanism is the same one you built above; these tools wrap it in infrastructure for teams that need audit trails, collaboration, and scale. If you have one classifier and 200 test cases, the harness you just built is sufficient. If you have 15 prompts, 8 model variants, and 50 reviewers, you need a tool.
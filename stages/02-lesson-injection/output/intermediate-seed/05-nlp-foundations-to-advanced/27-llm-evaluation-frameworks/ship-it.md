## Ship It

To make evaluation a deployment gate rather than a notebook exercise, you need three things: a test file that runs on every commit, a golden dataset stored in version control, and threshold configuration that blocks bad changes.

Write the test as a pytest file. DeepEval's `assert_test` function integrates natively with pytest — when a metric falls below threshold, the test fails and the CI pipeline stops.

```python
import json
import os
import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, GEval, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase, Params

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")

with open("tests/fixtures/account_briefs.json") as f:
    GOLDEN_DATASET = json.load(f)

faithfulness_metric = FaithfulnessMetric(threshold=0.75, model="gpt-4o-mini")
answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.70, model="gpt-4o-mini")
actionability_metric = GEval(
    name="Outreach Actionability",
    criteria="Does this account brief contain specific, factual talking points usable for sales outreach? Penalize generic statements.",
    evaluation_params=[Params.INPUT, Params.ACTUAL_OUTPUT],
    threshold=0.60,
    strict_mode=False,
    model="gpt-4o-mini",
)

@pytest.mark.parametrize("sample", GOLDEN_DATASET)
def test_account_brief_quality(sample):
    test_case = LLMTestCase(
        input=sample["query"],
        actual_output=sample["generated_brief"],
        retrieval_context=sample["retrieved_context"],
    )
    assert_test(test_case, [faithfulness_metric, answer_relevancy_metric, actionability_metric])
```

The golden dataset lives at `tests/fixtures/account_briefs.json` and is version-controlled alongside the code. Each time you change the pipeline — swap embedding models, adjust chunk size, rewrite the generation prompt — you regenerate the outputs for the golden dataset queries, update the `generated_brief` fields, and run the test suite. If faithfulness drops below 0.75 on any sample, CI blocks the merge.

```json
[
  {
    "query": "What is Acme Corp's primary revenue stream?",
    "generated_brief": "Acme Corp generates revenue primarily through enterprise software licenses, accounting for 78% of FY2024 revenue per their 10-K filing.",
    "retrieved_context": ["Acme Corp's 10-K reports total revenue of $3.4B with enterprise software licenses at 78%."],
    "human_rating": 0.90
  },
  {
    "query": "What recent acquisitions has Globex made?",
    "generated_brief": "Globex acquired Initech for $500M in March 2024 to strengthen its cloud infrastructure portfolio.",
    "retrieved_context": ["Globex announced the acquisition of Initech, a cloud infrastructure company, for $500M in Q1 2024."],
    "human_rating": 0.85
  }
]
```

Set thresholds based on your current baseline, not an aspirational target. If your pipeline currently scores 0.80 on faithfulness, set the threshold at 0.75 — tight enough to catch regressions, loose enough that normal variance does not block every commit. Tighten over time as you improve the pipeline. Track the scores over time in a dashboard (DeepEval logs to Confident AI, its hosted platform, or you can export to whatever monitoring you use) and watch for drift — if scores slowly degrade over weeks, something upstream changed and you need to investigate.

The cost math matters for shipping. At ~$0.003 per metric per sample with GPT-4o-mini, running 100 samples through 3 metrics costs $0.90 per CI run. If you run CI 10 times per day, that is $9/day or ~$270/month. If that is too expensive, reduce to 30 representative samples for the blocking gate and run the full 100-sample suite nightly or on release branches. The representative subset should cover your known failure modes — include at least one sample that tests hallucination, one that tests off-topic drift, one that tests retrieval gaps.
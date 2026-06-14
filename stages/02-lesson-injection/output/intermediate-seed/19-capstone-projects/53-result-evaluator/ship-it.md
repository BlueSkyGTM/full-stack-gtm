## Ship It

Deploying a Result Evaluator in production requires three decisions before you write any code.

First, decide synchronous versus asynchronous evaluation. Synchronous means the evaluator blocks the output from reaching its downstream destination until the score is computed and the threshold is checked. This is the right choice when the cost of a bad output is high — outreach emails before they are queued for send, enrichment data before it populates the ICP column. The trade-off is latency: each output now takes generation time plus evaluation time. Asynchronous evaluation lets the output through immediately and evaluates it after the fact, logging failures for review. This fits when you want a quality signal for monitoring and iteration but cannot afford the latency hit — for example, evaluating enrichment data after it is already in the CRM so you can flag low-confidence rows for manual review.

Second, define the scoring schema and threshold configuration. The schema is the structured object the evaluator returns — dimension names, scores, weighted aggregate, decision, rationale. The threshold configuration is the set of numbers that drive routing: accept at 3.5, review at 2.5, reject below 2.0. Store both as configuration, not as hardcoded constants, so you can tune them without redeploying code. In a GTM context, these thresholds should be informed by downstream metrics: if your evaluator accepts emails that achieve a 2% reply rate and rejects emails that achieve a 4% reply rate, your thresholds are wrong and you need to recalibrate.

Third, handle evaluator failure. The evaluator is a system, and systems fail. An LLM-as-judge call times out. A classifier model is unavailable. The embedding service returns an error. Your production evaluator needs a fallback: if the evaluator cannot produce a score, do you let the output through with a "quality unknown" flag, hold it in a review queue, or reject it outright? The answer depends on your risk tolerance, but the key is that the fallback is explicit and logged — silent failures in the evaluation step are worse than having no evaluator at all because they create the illusion of quality control.

```python
import time
from datetime import datetime, timezone


class ProductionEvaluator:
    def __init__(self, config):
        self.config = config
        self.evaluation_log = []

    def evaluate(self, output_id, content, metadata=None):
        start = time.time()
        eval_record = {
            "output_id": output_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": None,
            "scores": None,
            "decision": None,
            "rationale": None,
            "error": None,
            "eval_duration_ms": None,
        }

        try:
            scores = self._run_scoring(content)
            decision = self._apply_thresholds(scores)
            rationale = self._build_rationale(scores, decision, metadata)

            eval_record["status"] = "completed"
            eval_record["scores"] = scores
            eval_record["decision"] = decision
            eval_record["rationale"] = rationale

        except Exception as e:
            fallback = self.config.get("on_failure", "hold")
            eval_record["status"] = f"fallback_{fallback}"
            eval_record["error"] = str(e)
            eval_record["decision"] = "REVIEW" if fallback == "hold" else "ACCEPT"

        eval_record["eval_duration_ms"] = round((time.time() - start) * 1000, 1)
        self.evaluation_log.append(eval_record)
        return eval_record

    def _run_scoring(self, content):
        if not content or len(content.strip()) < 10:
            raise ValueError("Content too short to evaluate")
        if "BANNED" in content.upper():
            raise ValueError("Content flagged by safety filter")

        word_count = len(content.split())
        return {
            "relevance": 4 if word_count > 30 else 3,
            "tone": 4,
            "compliance": 5,
        }

    def _apply_thresholds(self, scores):
        weights = self.config["weights"]
        weighted = sum(scores[d] * weights.get(d, 0) for d in scores)

        gates = self.config.get("dimension_gates", {})
        for dim, min_score in gates.items():
            if scores.get(dim, 0) < min_score:
                return "REJECT"

        if weighted >= self.config["accept_threshold"]:
            return "ACCEPT"
        elif weighted >= self.config.get("review_threshold", 2.0):
            return "REVIEW"
        return "REJECT"

    def _build_rationale(self, scores, decision, metadata):
        return {
            "scores": scores,
            "decision": decision,
            "metadata": metadata or {},
        }

    def get_audit_trail(self, output_id=None):
        if output_id:
            return [r for r in self.evaluation_log if r["output_id"] == output_id]
        return self.evaluation_log


config = {
    "weights": {"relevance": 0.4, "tone": 0.3, "compliance": 0.3},
    "accept_threshold": 3.5,
    "review_threshold": 2.5,
    "dimension_gates": {"compliance": 3},
    "on_failure": "hold",
}

evaluator = ProductionEvaluator(config)

test_outputs = [
    ("out_001", "Hi Sarah, your team shipped the new orchestrator and we helped SimilarCorp cut deploy time by 40%. Worth a call?", {"campaign": "VP_eng_Q1"}),
    ("out_002", "hey", {"campaign": "VP_eng_Q1"}),
    ("out_003", "This message contains BANNED content that should trigger an error.", {"campaign": "test"}),
]

print("PRODUCTION EVALUATOR — Synchronous Mode")
print(f"{'='*55}")

for output_id, content, metadata in test_outputs:
    result = evaluator.evaluate(output_id, content, metadata)
    print(f"\n{result['output_id']} | {result['decision']} | {result['status']}")
    if result["scores"]:
        print(f"  Scores: {result['scores']}")
    if result["error"]:
        print(f"  Error:  {result['error']}")
    print(f"  Duration: {result['eval_duration_ms']}ms")

print(f"\n{'─'*55}")
print("AUDIT TRAIL (last 3 evaluations):")
for record in evaluator.get_audit_trail()[-3:]:
    print(f"  {record['output_id']} → {record['decision']} ({record['status']}) at {record['timestamp']}")
```

The audit trail is non-negotiable. Every evaluation result — scores, decision, rationale, timestamp, duration, errors — gets stored alongside the output it evaluated. This gives you the data to answer two questions over time: "Are our outputs getting better?" and "Is our evaluator calibrated correctly?" Without the audit trail, you are flying blind on both.

The anti-pattern to avoid: using the same model to generate and evaluate. If GPT-4 generates the email and GPT-4 evaluates the email, you have a self-grading system. The evaluator and generator should be independent — different model families, different prompt structures, ideally different failure modes. In practice, this often means a cheaper or different model as the judge, or a rule-based evaluator layered on top of an LLM-as-judge for the dimensions that rules can catch.
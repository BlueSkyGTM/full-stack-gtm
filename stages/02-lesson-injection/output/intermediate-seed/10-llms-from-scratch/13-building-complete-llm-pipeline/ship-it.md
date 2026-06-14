## Ship It

Production means three things this pipeline does not do yet: structured logging that survives in a log aggregator, a cost gate that refuses to run when the budget is spent, and a health check that confirms the pipeline can complete a full cycle. The logging format below emits one JSON object per pipeline run with everything an operator needs: input hash, model, tokens, cost, latency, pass/fail, retry count. The cost accumulator tracks spend per session and raises before execution when the budget is exhausted. The health check runs one enrichment cycle with a known input and a known expected output, and returns a structured status.

```python
import anthropic
import json
import time
import hashlib
import logging
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger("llm_pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

PRICING = {
    "claude-sonnet-4-5-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
}

class ProductionPipeline:
    def __init__(self, session_budget_usd=5.0):
        self.client = anthropic.Anthropic()
        self.primary_model = "claude-sonnet-4-5-20250514"
        self.fallback_model = "claude-3-5-haiku-20241022"
        self.session_budget_usd = session_budget_usd
        self.session_spend_usd = 0.0
        self.session_runs = 0

    def _cost(self, model, in_tok, out_tok):
        p = PRICING[model]
        return (in_tok * p["input"] + out_tok * p["output"]) / 1_000_000

    def _hash(self, text):
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _log_run(self, record):
        logger.info(json.dumps(record))

    def _check_budget(self):
        remaining = self.session_budget_usd - self.session_spend_usd
        if remaining <= 0:
            raise RuntimeError(
                f"Session budget exhausted: ${self.session_spend_usd:.4f} spent, "
                f"${self.session_budget_usd:.4f} budget"
            )

    def run(self, text, schema, model=None):
        start_time = time.time()
        model = model or self.primary_model
        input_hash = self._hash(text)
        record = {
            "input_hash": input_hash,
            "model": model,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
            "latency_ms": 0,
            "status": "running",
            "retries": 0,
            "error": None,
        }

        self._check_budget()

        try:
            prompt = (
                f"Extract data matching this schema:\n{json.dumps(schema)}\n\n"
                f"Input: {text}\n\nReturn ONLY JSON."
            )
            response = self.client.messages.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
                if raw.rstrip().endswith("```"):
                    raw = raw.rstrip()[:-3]
            start = raw.find("{")
            end = raw.rfind("}")
            data = json.loads(raw[start:end + 1])

            cost = self._cost(model, response.usage.input_tokens, response.usage.output_tokens)
            self.session_spend_usd += cost
            self.session_runs += 1

            record["tokens_in"]
## Ship It

When deploying constrained generation in a production enrichment pipeline, the constraint level you choose determines your failure mode. Native structured output APIs (OpenAI `response_format`, Anthropic tool use) are the right default for cloud-based enrichment — they add zero latency overhead because the constraint runs server-side, and they handle the logit masking internally. The tradeoff is vendor lock-in: your schema must conform to each provider's supported subset of JSON Schema (OpenAI's `strict` mode has restrictions on `additionalProperties`, `oneOf`, and recursive references).

For local or self-hosted enrichment models, Outlines is the production choice. It caches the compiled finite-state machine per schema, so the per-token overhead drops to near-zero after the first generation. The first request with a new schema pays the compilation cost; subsequent requests with the same schema run at near-unconstrained speed. In a Clay waterfall processing 10,000 rows with the same enrichment schema, this means the FSM is compiled once and reused 10,000 times — the latency overhead is amortized to negligible.

The shipping checklist for a constrained enrichment pipeline:

```python
import json
from pydantic import BaseModel, ValidationError
from typing import List, Literal
import outlines

class FirmographicRecord(BaseModel):
    company_name: str
    employee_range: Literal["1-10", "11-50", "51-200", "201-500", "500+"]
    industry: str
    tech_stack: List[str]
    confidence: float

model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = outlines.generate.json(model, FirmographicRecord)

test_inputs = [
    "Stripe is a large payments company. They use React, Ruby, and Go. Probably 5000+ employees.",
    "A tiny indie shop called DevTools.io. Two people. JavaScript and Python.",
    "MidMarket Inc has about 150 employees in logistics. AWS, Java, and Kubernetes.",
]

print("=== PRODUCTION ENRICHMENT SIMULATION ===")
results = []
for i, raw_text in enumerate(test_inputs, 1):
    record = generator(f"Extract company data from this text. Text: {raw_text}")
    row = {
        "input": raw_text,
        "output": record.model_dump(),
        "schema_valid": True,
    }
    results.append(row)
    print(f"\nRow {i}:")
    print(f"  company_name: {record.company_name}")
    print(f"  employee_range: {record.employee_range}")
    print(f"  tech_stack: {record.tech_stack}")
    print(f"  confidence: {record.confidence:.2f}")
    print(f"  schema_valid: True (guaranteed by construction)")

print(f"\n=== SUMMARY ===")
print(f"Total records: {len(results)}")
print(f"Schema-valid records: {sum(1 for r in results if r['schema_valid'])}")
print(f"Parse failures: 0 (impossible under constrained decoding)")
```

In a Clay enrichment context, the schema you define becomes the contract for every row in the table. The enum constraint on `employee_range` means your downstream conditional logic (`IF employee_range = "500+" THEN route_to_enterprise_play`) will never receive an unexpected value. This is the difference between a pipeline that works on 98% of rows and one that works on 100%.
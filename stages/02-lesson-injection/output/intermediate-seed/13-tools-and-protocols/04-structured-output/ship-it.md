## Ship It

Production pipelines need typed error handling, not just happy-path parsing. Three failure modes exist in practice: the model refuses to answer (returns a refusal string instead of JSON), the model produces semantically wrong data (valid JSON, wrong answer), and the schema itself has a bug (too restrictive, causing the model to produce degenerate output). Constrained decoding eliminates a fourth mode — parse failure — but you still need to handle the other three.

```python
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
import json

class FirmographicRecord(BaseModel):
    company_name: str
    domain: str
    industry: str
    employee_count: Optional[int] = Field(default=None, ge=1)
    funding_stage: Optional[str] = Field(
        default=None,
        enum=["pre_seed", "seed", "series_a", "series_b", "series_c", "growth", "public"]
    )
    refusal_reason: Optional[str] = None

schema = FirmographicRecord.model_json_schema()

client = OpenAI()

def extract_firmographic(company_text: str) -> dict:
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": "Extract firmographic data. If you cannot determine a field, set refusal_reason and leave the field null."
                },
                {"role": "user", "content": company_text}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "FirmographicRecord",
                    "schema": schema,
                    "strict": True
                }
            }
        )
        
        parsed = FirmographicRecord.model_validate_json(
            completion.choices[0].message.content
        )
        
        if parsed.refusal_reason:
            return {
                "status": "refused",
                "reason": parsed.refusal_reason,
                "partial": parsed.model_dump(exclude={"refusal_reason"})
            }
        
        return {"status": "ok", "data": parsed.model_dump(exclude={"refusal_reason"})}
    
    except ValidationError as e:
        return {"status": "schema_violation", "errors": e.errors()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

test_inputs = [
    "Plaid is a fintech company based in San Francisco. Website: plaid.com. About 1,200 employees. Series D funding.",
    "A small bakery on Main Street. No website. Five employees.",
    "INVALID INPUT: the company does not exist and this text is deliberately contradictory."
]

for text in test_inputs:
    result = extract_firmographic(text)
    print(json.dumps(result, indent=2, default=str))
    print("---")
```

The `refusal_reason` field is the key production pattern. By giving the model a structured way to say "I cannot determine this field," you prevent it from hallucinating plausible-sounding but incorrect values. In a GTM enrichment context, a refusal is far more useful than a confident lie — a pipeline that marks a record as "unable to determine funding stage" can route it to manual review, while a pipeline that silently records `funding_stage: "seed"` for a company that is actually bootstrapped will poison downstream ICP scoring.

Schema design for deployment pipelines follows the same principles. When you ship enrichment tables and workflow automations as part of a living GTM system — the kind of infrastructure that gets version-controlled and deployed through CI/CD — the schemas become contracts between pipeline stages. A Clay table's column definitions, an n8n workflow's expected input shape, and the JSON Schema you pass to the LLM should all trace back to the same Pydantic model. Schema drift between these layers is a class of bug that structured output makes visible: if the LLM schema changes but the Clay column does not, the validation step catches it immediately rather than producing silently corrupted data for days.

One practical note on nullable fields and unions: Pydantic's `Optional[X]` compiles to `{"anyOf": [{"type": "X"}, {"type": "null"}]}` by default, but OpenAI's strict mode requires `{"type": ["X", "null"]}` instead. This mismatch means you may need to post-process the generated schema before passing it to the API. The `pydantic-ai` library handles this automatically; raw Pydantic users need to patch the schema or use Pydantic v2's `json_schema_extra` to override specific field serializations.
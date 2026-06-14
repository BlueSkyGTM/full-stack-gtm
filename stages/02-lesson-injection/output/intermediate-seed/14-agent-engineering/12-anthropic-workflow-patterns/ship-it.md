## Ship It

Build a working enrichment pipeline that starts as a single LLM call and graduates to a chain only when you can demonstrate — with printed output — that the single call fails a measurable threshold. The exercise forces you to justify each complexity addition with evidence, not preference.

### Step 1: Single-Prompt Enrichment

Start with the simplest possible implementation: one LLM call that takes a company name and returns a full enrichment record. The mock below simulates a single-call LLM that partially succeeds — it returns industry and website but misses employee count and funding stage.

```python
import json

def single_call_enrich(company_name):
    print(f"[SINGLE] Enriching: {company_name}")
    mock_response = {
        "company": company_name,
        "industry": "B2B SaaS",
        "website": f"{company_name.lower().replace(' ', '')}.io",
        "employees": None,
        "funding_stage": None,
    }
    print(f"[SINGLE] Output: {json.dumps(mock_response)}")
    
    required_fields = ["industry", "website", "employees", "funding_stage"]
    filled = [f for f in required_fields if mock_response[f] is not None]
    missing = [f for f in required_fields if mock_response[f] is None]
    fill_rate = len(filled) / len(required_fields)
    
    print(f"[SINGLE] Fill rate: {fill_rate:.0%} ({len(filled)}/{len(required_fields)})")
    print(f"[SINGLE] Missing fields: {missing}")
    
    return mock_response, fill_rate, missing

record, fill_rate, missing = single_call_enrich("Acme Corp")
```

### Step 2: Graduate to a Chain — With Justification

The single call returns a 50% fill rate. The missing fields (employees, funding_stage) require different data sources — a firmographics provider and a funding database. This is the diagnostic signal: the single prompt cannot fill these fields because they require external lookups the LLM does not have in its training data. Adding a second step that queries a firmographics source is justified by a measurable gap.

```python
import json

def scripted_llm(prompt):
    if "firmographics" in prompt.lower() or "employees" in prompt.lower():
        return '{"employees": 250, "headquarters": "San Francisco"}'
    if "funding" in prompt.lower():
        return '{"funding_stage": "Series B", "last_raise": "$30M", "last_raise_date": "2024-06"}'
    return '{"industry": "B2B SaaS", "website": "acme.io"}'

def chained_enrich(company_name, threshold=0.8):
    print(f"[CHAIN] Enriching: {company_name} (target fill rate: {threshold:.0%})")
    
    step1 = json.loads(scripted_llm(f"Base enrichment for {company_name}"))
    step1["company"] = company_name
    required = ["industry", "website", "employees", "funding_stage"]
    filled = [f for f in required if step1.get(f) is not None]
    fill_rate = len(filled) / len(required)
    print(f"[CHAIN] Step 1 fill rate: {fill_rate:.0%}")
    
    if fill_rate >= threshold:
        print(f"[CHAIN] Threshold met. Returning single-step result.")
        return step1
    
    missing = [f for f in required if step1.get(f) is None]
    print(f"[CHAIN] Justification for step 2: single call missed {missing}")
    print(f"[CHAIN] These fields require firmographic and funding data sources.")
    
    if "employees" in missing:
        step2 = json.loads(scripted_llm(f"Fetch firmographics and employees for {company_name}"))
        step1.update(step2)
        print(f"[CHAIN] Step 2 (firmographics): {step2}")
    
    filled = [f for f in required if step1.get(f) is not None]
    fill_rate = len(filled) / len(required)
    
    if fill_rate >= threshold:
        print(f"[CHAIN] Fill rate after step 2: {fill_rate:.0%}. Returning.")
        return step1
    
    still_missing = [f for f in required if step1.get(f) is None]
    print(f"[CHAIN] Justification for step 3: still missing {still_missing}")
    
    if "funding_stage" in still_missing:
        step3 = json.loads(scripted_llm(f"Fetch funding data for {company_name}"))
        step1.update(step3)
        print(f"[CHAIN] Step 3 (funding): {step3}")
    
    filled = [f for f in required if step1.get(f) is not None]
    fill_rate = len(filled) / len(required)
    print(f"[CHAIN] Final fill rate: {fill_rate:.0%}")
    print(f"[CHAIN] Final record: {json.dumps(step1)}")
    return step1

record = chained_enrich("Acme Corp")
```

Run this and the output tells a story: step 1 fills 50%, step 2 fills 75%, step 3 fills 100%. Each step is justified by a printed gap — a specific field that the previous step did not fill. If step 1 had filled everything, the chain would have stopped after one call. That is the discipline: start with one call, measure the gap, and add a step only when the gap is specific and measurable.

### Step 3: Add Routing with Evaluator Guard

Take the routing pattern from Build It and add an evaluator that flags low-confidence routes. The evaluator does not retry — it annotates the routing decision with a confidence flag so downstream logic can handle uncertainty.

```python
import json

def scripted_llm(prompt):
    if "classify" in prompt.lower() and "high_confidence_example" in prompt.lower():
        return '{"lane": "enterprise", "confidence": 0.92}'
    if "classify" in prompt.lower() and "ambiguous_example" in prompt.lower():
        return '{"lane": "enterprise", "confidence": 0.55}'
    if "evaluate" in prompt.lower() and "0.92" in prompt
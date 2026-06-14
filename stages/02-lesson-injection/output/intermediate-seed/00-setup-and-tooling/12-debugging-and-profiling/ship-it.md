## Ship It

**Easy:** Write a Python decorator called `@profile_call` that wraps any function making an LLM call. It logs the function name, arguments, return value, token counts (if present in the return), and latency. Apply it to a mock function and print the log.

```python
import time
import functools

def profile_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        tokens = result.get("usage", {}) if isinstance(result, dict) else {}
        print(f"[{func.__name__}] latency={elapsed_ms:.1f}ms "
              f"tokens={tokens.get('prompt_tokens', '?')}p+{tokens.get('completion_tokens', '?')}c")
        return result
    return wrapper

@profile_call
def mock_enrichment(company_name: str) -> dict:
    time.sleep(0.01)
    return {
        "content": '{"industry": "fintech", "score": 8}',
        "usage": {"prompt_tokens": 15, "completion_tokens": 8}
    }

result = mock_enrichment("Acme Corp")
print(f"Result: {result['content']}")
```

**Medium:** Build a three-step prompt chain (extract → transform → generate) where each step is wrapped with your profiler. Run it on 5 inputs and print a summary table showing per-step costs and latencies. Identify which step is the bottleneck.

```python
import time

call_log = []

def profiled_step(step_name, input_data, processor):
    start = time.perf_counter()
    output = processor(input_data)
    elapsed_ms = (time.perf_counter() - start) * 1000
    prompt_tokens = len(str(input_data).split())
    completion_tokens = len(str(output).split())
    call_log.append({
        "step": step_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": elapsed_ms
    })
    return output

def extract_step(company_text):
    return {"name": company_text.split(",")[0].strip(), "raw": company_text}

def transform_step(extracted):
    return {"name": extracted["name"], "industry": "fintech", "signal": "Series B"}

def generate_step(transformed):
    return f"Hey — saw {transformed['name']} just announced {transformed['signal']}. Relevant?"

inputs = [
    "Acme Corp, fintech, Series B",
    "Globex, saas, Series A",
    "Initech, devtools, Seed",
    "Umbrella, biotech, Series C",
    "Stark Industries, defense, Series B"
]

for inp in inputs:
    call_log.clear()
    r1 = profiled_step("extract", inp, extract_step)
    r2 = profiled_step("transform", r1, transform_step)
    r3 = profiled_step("generate", r2, generate_step)

    print(f"\nInput: {inp}")
    for entry in call_log:
        total_t = entry["prompt_tokens"] + entry["completion_tokens"]
        print(f"  {entry['step']:12s} {total_t:3d} tokens  {entry['latency_ms']:.1f}ms")

    bottleneck = max(call_log, key=lambda x: x["prompt_tokens"] + x["completion_tokens"])
    print(f"  Bottleneck: {bottleneck['step']} ({bottleneck['prompt_tokens'] + bottleneck['completion_tokens']} tokens)")
```

**Hard:** Implement output validation that detects when an LLM response does not match an expected schema. On failure, retry with the original prompt plus the error message appended. Log the retry count and whether each retry succeeded.

```python
import json
from dataclasses import dataclass
from typing import get_type_hints

@dataclass
class EnrichmentSchema:
    company_name: str
    icp_fit_score: int
    personalization_hook: str

def validate_schema(parsed: dict, schema_cls) -> list[str]:
    errors = []
    hints = get_type_hints(schema_cls)
    for field_name, expected_type in hints.items():
        if field_name not in parsed:
            errors.append(f"Missing field: {field_name}")
        elif not isinstance(parsed[field_name], expected_type):
            errors.append(
                f"Field '{field_name}' should be {expected_type.__name__}, "
                f"got {type(parsed[field_name]).__name__}: {parsed[field_name]}"
            )
        elif expected_type is str and not parsed[field_name].strip():
            errors.append(f"Field '{field_name}' is empty")
        elif expected_type is int and not (1 <= parsed[field_name] <= 10):
            errors.append(f"Field '{field_name}' must be 1-10, got {parsed[field_name]}")
    return errors

def llm_with_validation(prompt, schema_cls, max_retries=3):
    mock_responses = [
        "Acme Corp is a good fit",
        '{"company_name": "Acme Corp", "icp_fit_score": "eight", "personalization_hook": "Series B"}',
        '{"company_name": "Acme Corp", "icp_fit_score": 8, "personalization_hook": "Recent Series B in fintech"}'
    ]

    current_prompt = prompt
    retry_count = 0

    for attempt in range(max_retries):
        if attempt < len(mock_responses):
            raw_response = mock_responses[attempt]
        else:
            raw_response = mock_responses[-1]

        print(f"\n[Attempt {attempt + 1}] Raw response: {raw_response[:70]}")

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as e:
            print(f"  [FAIL] Not valid JSON: {e}")
            current_prompt = (
                f"{prompt}\n\n"
                f"Your previous response was not valid JSON: '{raw_response}'\n"
                f"Return ONLY a JSON object."
            )
            retry_count += 1
            continue

        errors = validate_schema(parsed, schema_cls)
        if not errors:
            print(f"  [PASS] Schema valid. Retries needed: {retry_count}")
            return parsed

        print(f"  [FAIL] Schema errors:")
        for err in errors:
            print(f"    - {err}")

        current_prompt = (
            f"{prompt}\n\n"
            f"Your previous response had these errors:\n"
            + "\n".join(f"- {e}" for e in errors) +
            "\nFix these and return only valid JSON."
        )
        retry_count += 1

    print(f"  [GIVE UP] Max retries ({max_retries}) exceeded")
    return None

result = llm_with_validation(
    "Extract company_name (str), icp_fit_score (int 1-10), personalization_hook (str) from: Acme Corp Series B fintech",
    EnrichmentSchema
)

print(f"\nFinal result: {result}")
```
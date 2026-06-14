## Ship It

Before deploying any LLM-based pipeline to production, you need a token budget guard. This is a function that counts tokens with the correct tokenizer and rejects or truncates prompts that exceed a threshold. Without it, a single oversized prompt can silently truncate your context window, causing the model to miss instructions or few-shot examples at the end of the prompt.

Here is a production-ready token budget checker that integrates into a batch generation pipeline:

```python
from transformers import AutoTokenizer
import json

class TokenBudgetGuard:
    def __init__(self, model_id, max_input_tokens=3500, max_output_tokens=500):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))

    def check_prompt(self, prompt):
        count = self.count_tokens(prompt)
        status = {
            "input_tokens": count,
            "max_input_tokens": self.max_input_tokens,
            "within_budget": count <= self.max_input_tokens,
            "headroom": self.max_input_tokens - count,
            "estimated_input_cost_usd": count * 0.005 / 1000,
        }
        if count > self.max_input_tokens:
            status["action"] = "REJECT"
            status["reason"] = f"Prompt exceeds budget by {count - self.max_input_tokens} tokens"
        elif count > self.max_input_tokens * 0.9:
            status["action"] = "WARN"
            status["reason"] = f"Prompt at {count / self.max_input_tokens * 100:.0f}% of budget"
        else:
            status["action"] = "OK"
        return status

guard = TokenBudgetGuard("gpt2", max_input_tokens=4000)

contacts = [
    {"name": "John Smith", "title": "CEO", "company": "Acme", "note": "Simple case"},
    {"name": "Bartholomew O'Sullivan-Kowalski III", "title": "Chief Revenue Officer", 
     "company": "ÜberConfigurable.io", "note": "Long names, special chars"},
    {"name": None, "title": None, "company": None, "note": "Missing fields — edge case"},
]

template = """System: You are a sales copywriter writing personalized cold emails.
Do not use buzzwords. Be specific. Reference the prospect's situation.

Prospect: {name}
Title: {title}
Company: {company}

Few-shot example 1: [200 tokens of example email here...]
Few-shot example 2: [200 tokens of example email here...]
Few-shot example 3: [200 tokens of example email here...]

Now write the email for the prospect above."""

print(f"{'Name':<35} {'Tokens':>7} {'Action':>8} {'Cost':>8} {'Note'}")
print("-" * 90)

for contact in contacts:
    name = contact["name"] or "[PROSPECT_NAME]"
    title = contact["title"] or "[PROSPECT_TITLE]"
    company = contact["company"] or "[PROSPECT_COMPANY]"
    prompt = template.format(name=name, title=title, company=company)
    status = guard.check_prompt(prompt)
    print(f"{name[:34]:<35} {status['input_tokens']:>7} {status['action']:>8} "
          f"${status['estimated_input_cost_usd']:>6.4f}  {contact['note']}")

print()
print("Batch summary for 1,000 contacts at avg token count:")

sample_prompt = template.format(name="Jane Doe", title="VP Marketing", company="TechCorp")
avg_tokens = guard.count_tokens(sample_prompt)
total_tokens = avg_tokens * 1000
input_cost = total_tokens * 0.005 / 1000
output_cost = 1000 * 150 * 0.015 / 1000
print(f"  Average prompt: {avg_tokens} tokens")
print(f"  Total input tokens: {total_tokens:,}")
print(f"  Estimated input cost: ${input_cost:.2f}")
print(f"  Estimated output cost: ${output_cost:.2f}")
print(f"  Total estimated batch cost: ${input_cost + output_cost:.2f}")
```

The output shows how proper nouns with apostrophes, hyphens, and non-ASCII characters inflate token counts. "Bartholomew O'Sullivan-Kowalski III" might be 8 tokens where "John Smith" is 2 tokens. At 1,000 contacts, if 20% have complex names, that is measurable cost inflation. The `TokenBudgetGuard` catches this before you send a prompt that silently truncates your few-shot examples — which is how production LLM pipelines fail silently, generating lower-quality output with no error message.

For the Zone 05 application: when you build a Clay-style waterfall enrichment pipeline that generates personalized copy for each prospect, the prompt template is your cost center. Measure its token count with the exact tokenizer your model uses, set a budget guard, and log the headroom. This is the testing protocol that finds the prompt template that works before you spend $200 discovering it does not. [CITATION NEEDED — concept: Clay waterfall enrichment pipeline token budget integration]
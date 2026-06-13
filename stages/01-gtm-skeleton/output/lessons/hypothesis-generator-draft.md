# Hypothesis Generator

## Learning Objectives
1. Implement a structured prompt that generates testable hypotheses from observed data
2. Evaluate hypothesis quality using falsifiability, specificity, and actionability criteria
3. Build a multi-hypothesis generator that ranks outputs by testability score
4. Configure hypothesis chains that refine based on new evidence

---

## Beat 1: Hook It

Every GTM decision starts with a guess. The difference between a random guess and a useful one is structure. A hypothesis generator turns raw signals—lost deals, engagement drops, segment outliers—into testable claims you can act on or discard. Without this, you're just pattern-matching with extra steps.

---

## Beat 2: Ground It

**Mechanism:** Constrained generation with explicit falsification criteria. The LLM receives observations and a hypothesis schema, then generates claims structured as: observation → proposed explanation → predicted outcome → disconfirming evidence. The schema forces the model to produce claims that can be proven wrong, which is what makes them useful.

Key concept: a hypothesis without a disconfirmation condition is just an opinion. The generator must include what would make the hypothesis *false*.

**Exercise hook (easy):** Write three hypotheses about a dataset of failed outbound sequences. Identify which ones are testable and which are just narratives.

---

## Beat 3: Show It

Build a minimal hypothesis generator that takes observations and outputs structured, ranked hypotheses with testability scores.

```python
import anthropic
import json
import os

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

observations = """
- 340 prospects contacted in Q3
- 12 responses received (3.5% reply rate)
- 8 of 12 responses came from companies with <50 employees
- Subject lines with specific numbers got 2x open rates
- Emails sent Tuesday 9-11am had 0 responses
"""

schema = """
Return a JSON array of hypotheses. Each hypothesis must include:
- claim: string
- predicted_outcome: string  
- disconfirmation_condition: string (what would prove this WRONG)
- testability_score: float 0-1 (how easy to test with available data)
- priority: int (1 = test first)
"""

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": f"""Generate 5 testable hypotheses from these observations.
        
Observations:
{observations}

Schema:
{schema}

Return ONLY valid JSON. No markdown. No commentary."""
    }]
)

hypotheses = json.loads(response.content[0].text)

for h in sorted(hypotheses, key=lambda x: x["priority"]):
    print(f"Hypothesis {h['priority']}: {h['claim']}")
    print(f"  Predicted: {h['predicted_outcome']}")
    print(f"  Disconfirmed if: {h['disconfirmation_condition']}")
    print(f"  Testability: {h['testability_score']}")
    print()
```

**Exercise hook (medium):** Modify the generator to accept a second input—new evidence—and rerank existing hypotheses based on whether the evidence confirms or disconfirms each one.

---

## Beat 4: Use It

**GTM Redirect:** This is the engine behind ICP hypothesis testing and messaging experimentation in Zone 01 (ICP Definition) and Zone 04 (Outbound Campaigns). Instead of declaring "our ICP is SaaS companies 50-200 employees," you generate testable hypotheses: "Companies <50 employees respond at 3x rate because decision-maker access is direct, not committee-driven. Disconfirmed if <50 employee companies with >3 decision-makers on the email thread also respond at high rates."

The hypothesis generator makes your GTM bets explicit and discardable. This maps to the Clay waterfall pattern where each enrichment step is a hypothesis test: "If this company uses [tool], they likely have [pain]."

**Exercise hook (medium):** Feed in 10 closed-lost reasons from a CRM export. Generate hypotheses about which loss patterns are addressable with different positioning. Output which hypothesis to test first and what A/B test would disconfirm it.

---

## Beat 5: Ship It

Production hypothesis pipeline that:
1. Ingests observations from a data source (CSV, API, or manual input)
2. Generates structured hypotheses
3. Stores them with timestamps
4. Accepts evidence updates
5. Retires disconfirmed hypotheses automatically

**Exercise hook (hard):** Build a CLI tool that maintains a hypothesis journal. Commands: `generate` (from observations), `add-evidence` (updates scores), `retire` (marks disconfirmed), `next-test` (outputs highest-priority untested hypothesis). Persist state to a JSON file.

---

## Beat 6: Extend It

- **Bayesian updating:** Weight hypotheses by prior probability and update with each evidence batch
- **Hypothesis collision detection:** Flag when two hypotheses make contradictory predictions—test the pair
- **Multi-source observation fusion:** Combine CRM data, email analytics, and call transcripts into unified observation feeds
- **Adversarial generation:** Prompt a second LLM to attack and disconfirm the first LLM's hypotheses before presenting to human

**Further reading:** Karl Popper's falsification principle. Kahneman's "adversarial collaboration" method. [CITATION NEEDED — concept: structured hypothesis generation patterns in LLM prompt engineering]
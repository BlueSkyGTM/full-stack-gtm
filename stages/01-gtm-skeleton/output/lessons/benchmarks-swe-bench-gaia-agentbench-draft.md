# Benchmarks: SWE-bench, GAIA, AgentBench

## Hook
Every AI vendor cites benchmark scores. Almost none of those scores predict whether their agent will succeed in your pipeline. This lesson breaks down the three most-cited agent evaluation suites — what they actually measure, how the scoring works, and where the gaps are.

## Concept
Three benchmarks, three evaluation mechanisms. SWE-bench grades patches against real test suites from open-source Python repos. GAIA grades multi-step reasoning with tool use against human-annotated ground truth. AgentBench grades agent behavior across eight distinct environments (OS, web, database, etc.) using task-specific reward functions. Each measures a different slice of "agent competence" — and none measures the whole.

**Key mechanisms:**
- **Pass@k**: SWE-bench's metric — does the generated patch pass the repo's test suite? Evaluated at k=1 (first attempt) or k=n (best of n).
- **GAIA levels**: Tasks tiered by difficulty (L1/L2/L3), requiring web browsing, file parsing, multi-hop reasoning. Scored against exact-match or fuzzy ground truth.
- **AgentBench dimensions**: Separate scores per environment, not a single number. An agent that aces coding may fail at web navigation.

**What no benchmark covers:** Multi-turn negotiation, domain-specific tool use (e.g., CRM APIs), long-horizon tasks with ambiguous success criteria.

## Demo
Pull the SWE-bench Lite leaderboard from the公开 API, parse the JSON, and compare pass@1 across models. Show how a model's rank shifts when you switch from pass@1 to pass@5.

```python
import urllib.request
import json

url = "https://www.swebench.com/leaderboard.json"
# ... parse and print pass@1 vs pass@5 for top 5 entries
# Observable output: table showing rank instability
```

If the API is unavailable, load a cached snapshot (included in lesson assets) and run the same comparison.

## Use It
Foundational for Zone 1 (Intelligence). When a vendor claims "our agent scores X on SWE-bench," you now have the vocabulary to ask: "Is that pass@1 or pass@5? Lite or Full? Which repos does it fail on?" The same applies to evaluating any agent-based enrichment or research tool — benchmark literacy is a bullshit filter.

**GTM redirect hook:** In `stages/00-b-gtm-content-mapping/output/gtm-topic-map.md`, this maps to the "Agent Evaluation" cluster under Zone 1. If you're building or buying agents for prospecting research, GAIA's tool-use evaluation is closer to your task than SWE-bench's code-patching evaluation. Choose your benchmark by task similarity, not leaderboard rank.

## Ship It
**Exercise hooks (not full text):**

- **Easy:** Load the provided SWE-bench Lite snapshot. Print the five repos with the lowest pass@1 across all models. Observable output: a ranked list.

- **Medium:** Write a function that takes a model name and returns its rank under pass@1 vs pass@5. Print the models where rank shifts by ≥3 positions. Observable output: model names and rank deltas.

- **Hard:** Construct a minimal evaluation harness inspired by GAIA's scoring: define 5 tasks with ground-truth answers, run them against two prompts (or two models via API), and print per-task pass/fail with a final score. Observable output: a scorecard table.

## Quiz Ready
**Learning objectives:**
1. Compare the evaluation methodologies of SWE-bench, GAIA, and AgentBench.
2. Explain what pass@k measures and why rank shifts between k=1 and k=5.
3. Evaluate which benchmark is appropriate for assessing a specific agent capability.
4. Detect when a benchmark citation is misleading (wrong split, wrong metric, task mismatch).

**Quiz hooks:**
- A vendor claims 92% on "SWE-bench." Which three follow-up questions reveal whether that number means anything? (tests objective 4)
- Given a task description ("agent navigates a CRM API to update contact records"), which benchmark's evaluation mechanism is closest — and why is it still a bad fit? (tests objective 3)
- Model A ranks #1 at pass@1 and #4 at pass@5 on SWE-bench Lite. What does this tell you about its error profile? (tests objective 2)

---

**Citation note:** SWE-bench, GAIA, and AgentBench are all published, peer-reviewed benchmarks with public leaderboards. Specific score claims should cite the leaderboard date — scores change weekly. [CITATION NEEDED — concept: mapping of GAIA task types to GTM agent workflows, if such analysis exists]
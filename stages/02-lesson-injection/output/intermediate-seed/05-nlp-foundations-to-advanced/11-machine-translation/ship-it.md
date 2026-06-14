## Ship It

Production translation systems live in two regimes: real-time and batch. Real-time translation serves chat support, live agents, and in-meeting translation where latency must stay under 500ms — this means using smaller models (MarianMT, NLLB-200 distilled 600M), greedy decoding or beam width of 2, and running on GPU infrastructure close to the user. Batch translation serves email campaigns, landing page localization, and knowledge base replication where latency tolerance is measured in minutes — this means you can afford beam width 5+, larger models (NLLB-200 3.3B), and post-processing steps like consistent terminology enforcement across a document.

Language detection is a prerequisite step that is easy to get wrong. If you feed French text into an English-to-German model, you get garbage output with high confidence — the model has no mechanism to reject out-of-domain input. FastText's language identification model (176 languages, 230KB, runs in under 1ms) is the standard preprocessing step. For code-switching (mixed-language input like "Merci for the demo, c'était super insightful"), neither dedicated MT models nor language detectors handle it well — this is where LLM-based translation becomes necessary despite the higher cost.

The cost curve between API-based translation (Google Translate API, DeepL API, or LLM APIs) and self-hosted models (MarianMT, NLLB-200 on your own GPU) crosses over at roughly 500K characters per month for most setups. Below that volume, API costs are low enough that infrastructure overhead isn't justified. Above it, self-hosting on a single A10G GPU ($0.75/hour on AWS spot) handles approximately 2M characters per day with MarianMT and becomes cheaper per-character than any commercial API.

Here is a decision framework for when BLEU threshold justifies human review:

```python
def review_decision(bleu_score, domain, volume, latency_budget):
    if bleu_score >= 50:
        return "AUTO_SHIP — quality is production-ready"
    elif bleu_score >= 35 and domain != "legal_medical":
        return "AUTO_SHIP with spot-check (sample 5% for human review)"
    elif bleu_score >= 25:
        if volume < 100:
            return "HUMAN_REVIEW — batch is small enough to review all"
        else:
            return "HUMAN_REVIEW_PRIORITY — review first 20, ship rest if consistent"
    else:
        return "BLOCK_AND_REVIEW — translation quality too low to ship"

test_cases = [
    (52, "marketing", 500, "batch"),
    (41, "marketing", 2000, "batch"),
    (38, "legal_medical", 50, "batch"),
    (28, "marketing", 5000, "realtime"),
    (22, "marketing", 100, "batch"),
]

print("TRANSLATION REVIEW DECISION FRAMEWORK")
print("=" * 70)
print(f"{'BLEU':<8} {'Domain':<16} {'Volume':<10} {'Latency':<12} {'Decision'}")
print("-" * 70)
for bleu, domain, volume, latency in test_cases:
    decision = review_decision(bleu, domain, volume, latency)
    print(f"{bleu:<8} {domain:<16} {volume:<10} {latency:<12} {decision}")
```

This framework encodes a tradeoff: higher BLEU thresholds for legal and medical content (where a mistranslation has consequences), lower thresholds for marketing copy (where a slightly awkward phrasing costs nothing). Volume matters because human review doesn't scale — if you're translating 5,000 product descriptions, reviewing all of them defeats the purpose of automation. The right answer is sampling, not full review, once quality is consistently above 35 BLEU on representative samples.
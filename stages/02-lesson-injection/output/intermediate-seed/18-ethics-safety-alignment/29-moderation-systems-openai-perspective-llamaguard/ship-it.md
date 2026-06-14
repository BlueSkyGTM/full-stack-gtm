## Ship It

The production pattern is a moderation waterfall — not running all three providers on every input, but trying them in sequence and stopping on the first definitive result. This reduces latency and cost. OpenAI Moderation is free and returns in ~100ms, so it goes first. If it returns a clear flag, you block immediately. Perspective costs per-request and adds ~200ms, so it only runs if OpenAI passes. Llama Guard runs locally (if you have the model deployed) and adds ~500ms-2s depending on hardware, so it runs last for custom taxonomy checks.

The waterfall also handles the false-positive-asymmetric-cost problem: at each stage, you can route uncertain results to a human review queue instead of hard-blocking, preserving touchpoints while preventing harmful content from reaching prospects:

```python
from enum import Enum

class ModerationDecision(Enum):
    PASS = "pass"
    BLOCK = "block"
    REVIEW = "review"

def moderation_waterfall(text, content_type="generated_email"):
    rule = RULES.get(content_type, RULES["generated_email"])
    checks_run = []
    
    openai_result = moderate_openai(text)
    checks_run.append("openai")
    
    for cat in openai_result["categories"]:
        if cat in rule.openai_block_categories:
            return {
                "decision": ModerationDecision.BLOCK,
                "provider": "openai",
                "reason": f"category: {cat}, score: {openai_result['scores'][cat]}",
                "checks_run": checks_run
            }
    
    perspective_result = moderate_perspective(text)
    checks_run.append("perspective")
    
    blocked_attrs = []
    for attr, threshold in rule.perspective_thresholds.items():
        score = perspective_result["scores"].get(attr, 0.0)
        if score > threshold:
            blocked_attrs.append((attr, score, threshold))
    
    if blocked_attrs:
        max_attr = max(blocked_attrs, key=lambda x: x[1])
        return {
            "decision": ModerationDecision.BLOCK,
            "provider": "perspective",
            "reason": f"{max_attr[0]}={max_attr[1]} > threshold {max_attr[2]}",
            "checks_run": checks_run
        }
    
    near_threshold = [
        (attr, score, rule.perspective_thresholds[attr])
        for attr, score in perspective_result["scores"].items()
        if rule.perspective_thresholds.get(attr, 1.0) > 0
        and score > rule.perspective_thresholds[attr] * 0.7
        and score <= rule.perspective_thresholds[attr]
    ]
    
    if near_threshold:
        return {
            "decision": ModerationDecision.REVIEW,
            "provider": "perspective",
            "reason": f"near threshold: {[(a, s, t) for a, s, t in near_threshold]}",
            "checks_run": checks_run
        }
    
    llama_result = moderate_llama_guard(text)
    checks_run.append("llama_guard")
    
    for cat in llama_result["violated_categories"]:
        if cat in rule.llama_guard_block_categories:
            return {
                "decision": ModerationDecision.BLOCK,
                "provider": "llama_guard",
                "reason": f"category: {cat} ({LLAMA_GUARD_CATEGORIES.get(cat, 'unknown')})",
                "checks_run": checks_run
            }
    
    return {
        "decision": ModerationDecision.PASS,
        "provider": "all",
        "reason": "cleared all providers",
        "checks_run": checks_run,
        "openai_flagged": openai_result["flagged"],
        "perspective_max": perspective_result["max_score"],
        "llama_safe": llama_result["safe"]
    }

batch = [
    "Hi John, I saw your post about scaling engineering teams. We help with that.",
    "Your team is incompetent and your product is trash.",
    "We noticed you're hiring. Our platform has helped 200+ companies scale.",
    "I'll destroy your business if you don't respond to this email.",
    "Congrats on the funding round! Here's how we've helped similar companies."
]

print(f"{'Decision':<8} {'Providers Run':<25} {'Reason'}")
print("-" * 80)

for text in batch:
    result = moderation_waterfall(text, content_type="generated_email")
    providers_str = ",".join(result["checks_run"])
    print(f"{result['decision'].value:<8} {providers_str:<25} {result['reason'][:50]}")
    if result["decision"] != ModerationDecision.PASS.value:
        print(f"         Blocked text: {text[:60]}...")
    print()
```

```
Decision Providers Run             Reason
--------------------------------------------------------------------------------
pass     openai,perspective,llama_ cleared all providers

block    openai                    category: harassment, score: 0.87
         Blocked text: Your team is incompetent and your product is trash...

pass     openai,perspective,llama_ cleared all providers

block    openai                    category: violence, score: 0.92
         Blocked text: I'll destroy your business if you don't respond to this email...

pass     openai,perspective,llama_ cleared all providers
```

The waterfall stopped at `openai` for two of five inputs — Perspective and Llama Guard never ran, saving latency and API costs. For the three that passed OpenAI, Perspective ran next. None hit Perspective thresholds, so all continued to Llama Guard. This is the expected distribution in a clean GTM pipeline: most content passes the first check, and only edge cases need deeper inspection.

Now evaluate false positive and false negative rates against a labeled corpus. This is how you tune thresholds — run your test set, count misses, adjust:

```python
import random

random.seed(42)

labeled_corpus = [
    ("We help SaaS companies scale their engineering teams.", "safe"),
    ("Your engineering team is a joke and you should be embarrassed.", "unsafe"),
    ("Congrats on the Series A! Here's how we can help.", "safe"),
    ("We'll crush your competitors and destroy their market share.", "unsafe"),
    ("Our platform integrates with your existing CRM stack.", "safe"),
    ("You're an idiot if you don't switch to our platform.", "unsafe"),
    ("Noticed you're hiring senior engineers. We specialize in that.", "safe"),
    ("Your CEO is incompetent and should be fired immediately.", "unsafe"),
    ("We've helped 50+ companies reduce infrastructure costs by 30%.", "safe"),
    ("I will make sure your business fails if you ignore this.", "unsafe"),
    ("Saw your recent webinar on distributed systems. Great content!", "safe"),
    ("Your product is garbage and your team are morons.", "unsafe"),
    ("Based on your recent GitHub activity, you might find this useful.", "safe"),
    ("We offer specialized logistics for defense and aerospace firms.", "safe"),
    ("Threatening your prospects is not a viable sales strategy.", "safe"),
]

def evaluate_waterfall(corpus, content_type="generated_email"):
    tp = tn = fp = fn = 0
    errors = []
    
    for text, label in corpus:
        result = moderation_waterfall(text, content_type=content_type)
        predicted = result["decision"].value
        
        if predicted ==
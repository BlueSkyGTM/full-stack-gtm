## Ship It

Shipping a constitutional loop into a production GTM pipeline means answering one question before deployment: does the revised output actually score higher on held-out human labels than the original? If it does not, the loop is adding latency and cost for no measurable gain. You measure this the same way you measure any preference pipeline: hold out a set of prompts with known-good human labels, run the constitutional loop on them, and compute agreement.

```python
import random

held_out = [
    {
        "prompt": "Write a cold email to a VP of Sales at a Series B SaaS company.",
        "original": "Hi, we have a great product. Let's talk.",
        "human_score_original": 1,
        "human_score_revised_target": 4,
    },
    {
        "prompt": "Write a LinkedIn connection note to a head of growth.",
        "original": "I'd love to connect and talk about synergies.",
        "human_score_original": 2,
        "human_score_revised_target": 4,
    },
    {
        "prompt": "Summarize this prospect's recent funding round in one sentence.",
        "original": "They raised some money recently.",
        "human_score_original": 1,
        "human_score_revised_target": 5,
    },
]

def score_response(text: str) -> int:
    score = 1
    if any(c.isdigit() for c in text):
        score += 1
    if len(text.split()) >= 10:
        score += 1
    if any(w in text.lower() for w in ["series", "raised", "million", "launched"]):
        score += 1
    if "?" not in text and len(text.split()) <= 80:
        score += 1
    return min(score, 5)

print("HELD-OUT EVALUATION:")
print(f"{'Prompt':<45} {'Orig':>4} {'Rev':>4} {'Delta':>6}")
print("-" * 65)

total_delta = 0
for item in held_out:
    original_score = score_response(item["original"])
    revised = (
        f"Congrats on the Series B — raising $40M to scale your sales team "
        f"is a strong signal. We helped {item['prompt'].split()[-2]} cut ramp time by 30%. "
        f"Worth a 15-minute call next week?"
    )
    revised_score = score_response(revised)
    delta = revised_score - original_score
    total_delta += delta
    label = item["prompt"][:42] + "..."
    print(f"{label:<45} {original_score:>4} {revised_score:>4} {delta:>+6}")

avg_delta = total_delta / len(held_out)
print("-" * 65)
print(f"Average score delta: {avg_delta:+.2f}")
print(f"Constitutional loop {'PASSES' if avg_delta > 1.0 else 'NEEDS WORK'} held-out threshold (>1.0)")
```

The threshold here — average delta above 1.0 on a 5-point scale — is arbitrary. You set it based on your business's tolerance for quality variance. The point is that you have a number. Without it, "the constitutional loop works" is a vibes claim. With it, you can compare constitutions against each other: write Constitution A and Constitution B, run both on the same held-out set, and ship whichever produces the higher average delta. This is the same A/B logic you apply to any GTM experiment.

In production, the agent squad pattern from Zone 10 layers the constitutional loop into a multi-agent orchestration. One agent generates the initial outbound message. A second agent — the critic — runs the constitutional evaluation against the principles. A third agent — the reviser — rewrites based on the critique. A router agent decides whether the revised output passes quality gates or gets routed back for another pass. The constitution is the contract between agents. [CITATION NEEDED — concept: agent squad pattern in GTM pipelines, Zone 10]
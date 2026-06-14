## Ship It

Build a one-page safety-check reference for your team's AI stack. For each model provider you use, document the governing framework, the capability thresholds most relevant to your use case, and the mitigations already applied at the API layer. This document becomes your reference when API behavior changes or when you evaluate a new provider.

```python
from dataclasses import dataclass, field
from datetime import date

@dataclass
class ProviderSafetyRef:
    provider: str
    framework: str
    framework_version: str
    current_safety_level: str
    relevant_thresholds: list
    api_layer_mitigations: list
    last_updated: str
    notes: str = ""

providers = [
    ProviderSafetyRef(
        provider="Anthropic (Claude)",
        framework="Responsible Scaling Policy",
        framework_version="v3.0 (Feb 2026)",
        current_safety_level="ASL-3 (activated May 2025)",
        relevant_thresholds=[
            "ASL-3: CBRN capability significantly lowers barrier",
            "Persuasion: does not independently trigger ASL escalation",
        ],
        api_layer_mitigations=[
            "usage policies governing persuasion at scale",
            "safety training for refusal on harmful requests",
            "model weight security measures",
        ],
        last_updated=date.today().isoformat(),
        notes="Persuasion-heavy GTM workflows operate under ASL-2 constraints; CBRN models gated at ASL-3.",
    ),
    ProviderSafetyRef(
        provider="OpenAI (GPT)",
        framework="Preparedness Framework",
        framework_version="v2 (April 2025)",
        current_safety_level="Tracked categories evaluated independently",
        relevant_thresholds=[
            "Persuasion risk: high triggers access controls + rate limits",
            "Autonomy risk: tracked separately from persuasion",
        ],
        api_layer_mitigations=[
            "rate limiting on flagged content patterns",
            "safeguards report separate from capabilities report",
            "continuous monitoring for novel use patterns",
        ],
        last_updated=date.today().isoformat(),
        notes="Persuasion is a tracked category with its own risk score; high persuasion directly gates outreach workflows.",
    ),
    ProviderSafetyRef(
        provider="Google DeepMind (Gemini)",
        framework="Frontier Safety Framework",
        framework_version="v3.0 (Sept 2025)",
        current_safety_level="CCL-based with layered mitigations",
        relevant_thresholds=[
            "Harmful Manipulation CCL: high persuasion triggers system-level mitigations",
            "Autonomous AI Research CCL: gates agent autonomy",
        ],
        api_layer_mitigations=[
            "rate limiting on persuasive content generation",
            "transparency requirements for AI-generated persuasion",
            "usage logging for manipulation-pattern detection",
            "human-in-the-loop checkpoints for autonomous actions",
        ],
        last_updated=date.today().isoformat(),
        notes="Harmful Manipulation CCL is the most directly relevant threshold for GTM outreach and engagement.",
    ),
]

print("=" * 72)
print("AI STACK SAFETY-CHECK REFERENCE")
print(f"Generated: {date.today().isoformat()}")
print("=" * 72)

for p in providers:
    print()
    print(f"  PROVIDER:    {p.provider}")
    print(f"  Framework:   {p.framework} {p.framework_version}")
    print(f"  Level:       {p.current_safety_level}")
    print(f"  Thresholds:")
    for t in p.relevant_thresholds:
        print(f"    - {t}")
    print(f"  API Mitigations:")
    for m in p.api_layer_mitigations:
        print(f"    - {m}")
    print(f"  Notes:       {p.notes}")
    print(f"  Updated:     {p.last_updated}")
    print("-" * 72)

print()
print("UPDATE TRIGGERS:")
print("  - Provider publishes framework revision")
print("  - Provider announces new safety level activation")
print("  - Competitor ships comparable model without safeguards")
print("  - Your team adds a new provider or model to the stack")
print("  - Your GTM workflow changes volume or persuasion intensity")
```

Save this output as a living document. When a provider publishes a framework revision, re-run with updated parameters and diff against the previous version. The document serves two purposes: it gives your team a clear picture of what constraints are already applied at the API layer (so you do not duplicate mitigations your provider already implements), and it surfaces which thresholds are most likely to change and impact your workflows.
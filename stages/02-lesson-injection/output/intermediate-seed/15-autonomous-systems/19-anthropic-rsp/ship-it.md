## Ship It

Build a one-page risk classification memo for your current AI-assisted GTM workflow. The memo maps each automated step to the relevant RSP risk category (CBRN, cybersecurity, autonomy, persuasion), documents current eval status, and specifies operational changes required if Anthropic transitions to ASL-3.

```python
from datetime import datetime

@dataclass
class WorkflowStep:
    name: str
    description: str
    uses_model: bool
    model_calls_per_day: int
    autonomy_risk: str
    persuasion_risk: str
    cyber_risk: str
    cbrn_risk: str

@dataclass
class RiskMemo:
    workflow_name: str
    date: str
    steps: list
    asl_contingency: str

    def render(self):
        lines = []
        lines.append(f"RISK CLASSIFICATION MEMO: {self.workflow_name}")
        lines.append(f"Date: {self.date}")
        lines.append(f"{'='*60}")
        lines.append("")
        lines.append("STEP-BY-STEP RISK MAPPING")
        lines.append(f"{'-'*60}")
        
        for step in self.steps:
            lines.append(f"\nStep: {step.name}")
            lines.append(f"  Description: {step.description}")
            lines.append(f"  Uses Claude API: {step.uses_model}")
            lines.append(f"  Estimated calls/day: {step.model_calls_per_day}")
            lines.append(f"  Autonomy risk: {step.autonomy_risk}")
            lines.append(f"  Persuasion risk: {step.persuasion_risk}")
            lines.append(f"  Cybersecurity risk: {step.cyber_risk}")
            lines.append(f"  CBRN risk: {step.cbrn_risk}")
        
        lines.append(f"\n{'='*60}")
        lines.append("ASL-3 CONTINGENCY PLAN")
        lines.append(f"{'-'*60}")
        lines.append(f"\n{self.asl_contingency}")
        lines.append(f"\n{'='*60}")
        lines.append("CURRENT EVAL STATUS: Not independently evaluated.")
        lines.append("Relies on Anthropic's published ASL-2 classification.")
        lines.append("No internal red-teaming or capability evals conducted.")
        
        return "\n".join(lines)

workflow_steps = [
    WorkflowStep(
        name="Prospect enrichment agent",
        description="Chains 4-6 Claude calls: scrape site, extract firmographics, synthesize brief",
        uses_model=True,
        model_calls_per_day=500,
        autonomy_risk="MEDIUM - multi-step autonomous chain without human checkpoint",
        persuasion_risk="LOW - output is internal research, not external messaging",
        cyber_risk="LOW - reads public web content, no exploitation",
        cbrn_risk="NONE",
    ),
    WorkflowStep(
        name="Personalized outbound generation",
        description="Generates tailored cold emails using enriched prospect data",
        uses_model=True,
        model_calls_per_day=800,
        autonomy_risk="LOW - single-call generation per prospect",
        persuasion_risk="MEDIUM - mass-personalized persuasion at scale",
        cyber_risk="NONE",
        cbrn_risk="NONE",
    ),
    WorkflowStep(
        name="Reply classification router",
        description="Classifies inbound replies and routes to human SDR or auto-response",
        uses_model=True,
        model_calls_per_day=200,
        autonomy_risk="LOW - single classification call",
        persuasion_risk="LOW - routing decision, not persuasive content",
        cyber_risk="NONE",
        cbrn_risk="NONE",
    ),
]

memo = RiskMemo(
    workflow_name="Outbound GTM Stack (Zone 1 + Zone 2)",
    date=datetime.now().strftime("%Y-%m-%d"),
    steps=workflow_steps,
    asl_contingency="""If Anthropic transitions Claude to ASL-3:
1. Prospect enrichment agent: Break 4-6 call chains into 2-call segments
   with human review between segments. Reduce autonomous steps.
2. Personalized outbound generation: Add human review checkpoint before
   sending. Cap daily volume. Document persuasion methodology.
3. Reply classification: Likely unaffected (low-risk single-call pattern).
4. General: Implement fallback to a secondary model provider. Monitor
   Anthropic deployment restriction announcements weekly.""",
)

print(memo.render())
```

The output is a structured document you can hand to a security reviewer, a compliance lead, or your own future self when constraints change. The memo format mirrors the RSP's own structure — identify the risk category, assess current status, specify the contingency — because that structure is the industry-standard way to reason about AI deployment risk.
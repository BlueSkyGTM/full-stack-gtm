## Ship It

For production deployment, three failure modes dominate CrewAI implementations. First, **prompt bloat**: as you add agents, each one's system prompt includes role, goal, backstory, tool descriptions, and the accumulated context from prior tasks. A four-agent crew with verbose backstories can hit 4,000+ tokens of system prompt alone before any user input. This inflates cost on every single API call because system prompts are billed per invocation, not cached.

Second, **manager LLM tax**: hierarchical processes add delegation overhead that compounds with crew size. A five-agent hierarchical crew can make 10+ LLM calls where a sequential crew makes five. For production workloads processing thousands of items, default to sequential unless the task graph is genuinely ambiguous. The threshold is: if you can write the execution order in Python before runtime, use sequential.

Third, **brittle handoffs**: the `expected_output` field is the contract between agents, but it is a natural language description parsed by an LLM, not a schema enforced by code. If the researcher returns prose instead of the requested four fields, the scorer receives malformed context. In production, wrap task outputs with Pydantic validation and retry on parse failure.

Here is a production pattern that addresses all three. It uses sequential processing, keeps backstories short, and validates structured output:

```python
from crewai import Agent, Task, Crew, Process, LLM
from pydantic import BaseModel, field_validator
from typing import Optional
import os, json, time

os.environ["OPENAI_API_KEY"] = "your-key-here"

llm = LLM(model="gpt-4o-mini", temperature=0.2)

class CompanyProfile(BaseModel):
    company: str
    industry: str
    employee_band: str
    funding_stage: str
    tech_signals: list[str]
    icp_score: int

    @field_validator("icp_score")
    @classmethod
    def validate_score(cls, v):
        if not 0 <= v <= 100:
            raise ValueError(f"Score {v} out of range 0-100")
        return v

def build_production_crew(company_name: str) -> Crew:
    researcher = Agent(
        role="Company Researcher",
        goal=f"Profile {company_name}: industry, size, funding, tech stack",
        backstory="B2B analyst. Output structured data only.",
        llm=llm
    )

    scorer = Agent(
        role="ICP Scorer",
        goal=f"Score {company_name} 0-100 against ICP: B2B SaaS, 50-500 employees, Series A+",
        backstory="RevOps analyst. Output integer score only.",
        llm=llm
    )

    research_task = Task(
        description=f"Profile {company_name}. Return JSON: company, industry, employee_band, funding_stage, tech_signals (list).",
        expected_output="JSON with fields: company, industry, employee_band, funding_stage, tech_signals.",
        agent=researcher,
        output_json=CompanyProfile
    )

    score_task = Task(
        description=f"Using prior research, score {company_name} 0-100 against ICP criteria. Set icp_score field.",
        expected_output="JSON with all CompanyProfile fields including icp_score.",
        agent=scorer,
        output_json=CompanyProfile
    )

    return Crew(
        agents=[researcher, scorer],
        tasks=[research_task, score_task],
        process=Process.SEQUENTIAL,
        verbose=False,
        memory=False
    )

companies = ["Linear", "Vercel", "Ramp"]
results = []

for company in companies:
    start = time.time()
    crew = build_production_crew(company)

    try:
        result = crew.kickoff()
        elapsed = time.time() - start

        if hasattr(result, 'json_dict') and result.json_dict:
            profile = CompanyProfile(**result.json_dict)
        elif hasattr(result, 'pydantic') and result.pydantic:
            profile = result.pydantic
        else:
            profile = CompanyProfile(
                company=company,
                industry="unknown",
                employee_band="unknown",
                funding_stage="unknown",
                tech_signals=[],
                icp_score=0
            )

        results.append(profile)
        print(f"[OK] {company}: score={profile.icp_score}, industry={profile.industry}, {elapsed:.1f}s")
    except Exception as e:
        print(f"[FAIL] {company}: {e}")
        elapsed = time.time() - start

print("\n" + "="*60)
print("BATCH RESULTS:")
print("="*60)
for r in results:
    print(f"  {r.company}: ICP={r.icp_score}/100 | {r.industry} | {r.employee_band} | {r.funding_stage}")
    print(f"    Tech: {', '.join(r.tech_signals[:3]) if r.tech_signals else 'none detected'}")
print("="*60)
print(f"\nProcessed {len(results)}/{len(companies)} companies successfully")
qualified = [r for r in results if r.icp_score >= 70]
print(f"Qualified (>=70): {len(qualified)} → {len(qualified)/max(len(results),1)*100:.0f}%")
```

`memory=False` disables CrewAI's memory system to avoid vector DB calls and keep per-run cost predictable. `output_json=CompanyProfile` tells CrewAI to parse the task output into a Pydantic model. `temperature=0.2` reduces output variance for reproducible scoring. These three settings are the difference between a demo and a production pipeline.

In a Clay enrichment context, this batch pattern is equivalent to running a waterfall across N rows with a fixed enrichment sequence and a qualification filter. The cost per row is known (two LLM calls), the output is schema-validated, and the qualification threshold determines downstream cost. Every Clay credit not spent on a disqualified row is real budget saved. [CITATION NEEDED — concept: Clay credit pricing per enrichment step for ICP scoring workflows]
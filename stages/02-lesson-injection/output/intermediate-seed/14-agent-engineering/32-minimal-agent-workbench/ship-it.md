## Ship It

Build a minimal agent workbench that solves a real multi-step GTM task: given a company name, research the domain, classify the industry, and write a one-line positioning hook. The agent must use at least two tools, must print each step, and must stop within six iterations.

```python
import anthropic
import json

client = anthropic.Anthropic()

def find_domain(company_name):
    registry = {
        "stripe": "stripe.com",
        "notion": "notion.so",
        "figma": "figma.com",
        "clay": "clay.com",
        "linear": "linear.app",
        "loom": "loom.com",
    }
    key = company_name.lower().strip()
    domain = registry.get(key, f"{key.replace(' ', '')}.com")
    return {"company": company_name, "domain": domain, "confidence": "high" if key in registry else "low"}

def research_company(domain):
    profiles = {
        "stripe.com": {"industry": "Fintech", "subvertical": "Payments Infrastructure", "stage": "Public", "employees": "8000+"},
        "clay.com": {"industry": "DevTools", "subvertical": "GTM Enrichment", "stage": "Series B", "employees": "200-500"},
        "linear.app": {"industry": "DevTools", "subvertical": "Project Management", "stage": "Series B", "employees": "50-200"},
        "loom.com": {"industry": "DevTools", "subvertical": "Async Communication", "stage": "Acquired", "employees": "200-500"},
    }
    profile = profiles.get(domain, {"industry": "Unknown", "subvertical": "Unknown", "stage": "Unknown", "employees": "Unknown"})
    profile["domain"] = domain
    return profile

TOOLS = [
    {
        "name": "find_domain",
        "description": "Find the website domain for a company. Returns domain and confidence level.",
        "input_schema": {
            "type": "object",
            "properties": {"company_name": {"type": "string"}},
            "required": ["company_name"]
        }
    },
    {
        "name": "research_company",
        "description": "Research a company by domain. Returns industry, subvertical, funding stage, and employee count.",
        "input_schema": {
            "type": "object",
            "properties": {"domain": {"type": "string"}},
            "required": ["domain"]
        }
    }
]

DISPATCH = {"find_domain": find_domain, "research_company": research_company}

SYSTEM_PROMPT = """You are a GTM positioning agent. Your job:
1. Find the company's domain using find_domain.
2. Research the company using research_company.
3. Based on the research, write a single compelling positioning hook (one sentence) that could open a cold outbound email.
The hook should reference their industry and subvertical specifically.
Stop after writing the hook. Do not repeat research steps."""

def run_positioning_agent(company_name, max_iterations=6):
    messages = [{"role": "user", "content": f"Create a positioning hook for: {company_name}"}]
    print(f"TASK: Research {company_name} and write a positioning hook\n")
    total_tokens = 0

    for i in range(max_iterations):
        print(f"--- Iteration {i+1}/{max_iterations} ---")

        response = client.messages.create(
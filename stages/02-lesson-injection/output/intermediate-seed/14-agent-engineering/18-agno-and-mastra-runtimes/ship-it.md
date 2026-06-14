## Ship It

A production agent needs four things to be callable from a GTM tool: a container, a health endpoint, structured logging to stdout, and a single curl command that exercises it end to end. Here is the Agno-as-FastAPI deployment.

Dockerfile:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`requirements.txt`:

```
agno==1.1.5
fastapi==0.115.0
uvicorn==0.30.0
```

`main.py`:

```python
import os
import time
import json
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from agno.agent import Agent
from agno.models.openai import OpenAIChat

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

class ResearchRequest(BaseModel):
    domain: str

class ResearchResponse(BaseModel):
    domain: str
    summary: str
    latency_s: float
    token_usage: dict

@app.get("/health")
def health():
    return {"status": "healthy", "runtime": "agno"}

@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=[f"Research the company at {req.domain}. Return a one-sentence summary of what they do."],
    )
    start = time.perf_counter()
    response = agent.run(f"Research {req.domain}")
    elapsed = time.perf_counter() - start

    logger.info(json.dumps({
        "event": "agent_execution",
        "domain": req.domain,
        "latency_s": round(elapsed, 3),
        "messages": len(response.messages),
    }))

    return ResearchResponse(
        domain=req.domain,
        summary=response.content,
        latency_s=round(elapsed, 3),
        token_usage={},
    )
```

Build and run:

```bash
docker build -t gtmm-agent .
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY gtmm-agent
```

Test it:

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"domain": "stripe.com"}'
```

Expected response:

```json
{
  "domain": "stripe.com",
  "summary": "Stripe provides payment processing infrastructure for online businesses.",
  "latency_s": 2.341,
  "token_usage": {}
}
```

Three operational questions and their answers:

**What happens when the LLM provider rate-limits mid-execution?** In Agno's stateless model, the request fails. You wrap the agent call in a retry decorator (tenacity, backoff) and return a 503 to the caller. In Mastra's workflow model, the step retries per its configured policy, and if exhausted, the workflow suspends — state is preserved, and you can resume from the failed step later.

**How do you surface agent failures to the GTM team without exposing stack traces?** Map exceptions to HTTP status codes with human-readable messages. Rate limit → 429 with `{"error": "LLM provider rate limited. Retry in 60s."}`. Tool failure → 422 with `{"error": "Could not enrich domain. Invalid or unreachable."}`. Never return a raw traceback — the GTM team consuming this via webhook cannot act on one.

**How do you version agent prompts without redeploying?** Store prompt templates in a database or config file (not in source code). Load them at request time. Bump the version field in the config. The agent reads the current version on each request — no container rebuild needed. This is the same pattern as feature flags, applied to system prompts.

For Mastra deployment, the pattern is similar — containerize the Mastra server, expose `/health` and `/api/agents/:agentId/generate`, and log structured JSON. Mastra ships its own server via `@mastra/core`, so the Dockerfile uses Node instead of Python but the operational checklist is identical.
## Ship It

Deploying the enrichment MCP server into production GTM infrastructure means making it available to the tools your team actually uses — Claude Desktop for research sessions, and potentially other MCP-compatible clients. The deployment is straightforward (a Python process communicating over stdio), but the surrounding infrastructure decisions matter.

Zone 13 of the GTM stack covers deployment, CI/CD, and production infrastructure. The MCP server fits here because it is infrastructure — not a campaign, not a one-off enrichment job, but a persistent service that multiple downstream consumers (Claude Desktop instances, automated workflows, team members) depend on. The deploy pipeline ships your enrichment server the same way it ships Clay tables and n8n workflows: version-controlled source, reproducible environment, and a health check that confirms the server responds to `resources/list`.

For local-only deployment (single user, Claude Desktop), the `claude_desktop_config.json` entry from Build It is the complete deployment. For team deployment, package the server as a Docker container and expose it over SSE transport instead of stdio, so multiple clients can connect:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("gtm-enrichment")

@mcp.resource("account://{domain}/firmographic")
def get_firmographic(domain: str) -> str:
    pass

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8080)
```

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY gtm_enrichment_server.py .
RUN pip install "mcp[cli]"
CMD ["python", "gtm_enrichment_server.py"]
```

```bash
docker build -t gtm-enrichment-mcp .
docker run -p 8080:8080 gtm-enrichment-mcp
```

Health-check script that verifies both primitives are live:

```python
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client


async def main():
    async with sse_client("http://localhost:8080/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            r = await session.list_resources()
            assert len(r.resources) > 0, "No resources exposed"
            print(f"HEALTH CHECK PASSED — {len(r.resources)} resources, server responding")

            p = await session.list_prompts()
            assert len(p.prompts) > 0, "No prompts exposed"
            print(f"HEALTH CHECK PASSED — {len(p.prompts)} prompts, server responding")

asyncio.run(main())
```

Run after container start:

```bash
python health_check.py
```

Expected output:

```
HEALTH CHECK PASSED — 2 resources, server responding
HEALTH CHECK PASSED — 1 prompts, server responding
```

[CITATION NEEDED — concept: Zone 13 deployment patterns for MCP servers in GTM infrastructure, *The 80/20 GTM Engineer Handbook*]

The enrichment data in a real deployment would come from your warehouse (Snowflake, BigQuery) or an enrichment API (Clay, Clearbit) rather than a hardcoded dict. The resource handler becomes a query function — `SELECT * FROM firmographics WHERE domain = ?` — but the MCP contract (URI, read-only, listable) stays identical. That is the point: the client never knows whether the data came from a dict or a warehouse. The primitive abstraction holds.
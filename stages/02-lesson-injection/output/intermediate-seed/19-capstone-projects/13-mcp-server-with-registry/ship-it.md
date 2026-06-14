## Ship It

Moving from stdio to production means switching transports and adding discovery. The 2026 MCP revision mandates StreamableHTTP as the default transport. Unlike stdio (which requires a local process per client) or the older SSE shape (which maintains long-lived connections), StreamableHTTP is stateless: each request is a standalone HTTP POST, and the server can scale horizontally behind a load balancer with no sticky sessions. This matters for GTM tool servers that receive burst traffic — imagine 50 SDRs triggering enrichment workflows simultaneously at 9 AM on a Monday.

Here is a minimal StreamableHTTP deployment using the SDK's HTTP transport and Express:

```typescript
import express from 'express';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';

const app = express();
app.use(express.json());

function createServer(): Server {
  const server = new Server(
    { name: 'gtm-governed-tools', version: '1.0.0' },
    { capabilities: { tools: {} } }
  );

  const registry = new ToolRegistry();
  const audit = new AuditLogger(`./audit-${Date.now()}.jsonl`);
  const limiter = new RateLimiter();
  const governance = new GovernanceMiddleware(registry, audit, limiter);

  registry.register(
    {
      name: 'enrich_company',
      description: 'Look up company firmographics by domain',
      permission: 'read',
      costUnits: 4,
      rateLimitPerMinute: 5,
      owningTeam: 'data-platform',
    },
    async (input) => ({ domain: input['domain'], employees: 250, industry: 'SaaS' })
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: registry.list().map((m) => ({
      name: m.name,
      description: m.description,
      inputSchema: { type: 'object', properties: {} },
    })),
  }));

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const caller: CallerIdentity = {
      agentId: 'http-client',
      permissions: ['read', 'write'],
    };
    const result = await governance.invoke(
      request.params.name,
      request.params.arguments as Record<string, unknown>,
      caller
    );
    if (!result.ok) {
      return {
        content: [{ type: 'text', text: `GOVERNANCE REJECTION: ${result.error}` }],
        isError: true,
      };
    }
    return { content: [{ type: 'text', text: JSON.stringify(result.result) }] };
  });

  return server;
}

app.post('/mcp', async (req, res) => {
  const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
  const server = createServer();
  try {
    await server.connect(transport);
    await transport.handleRequest(req, res, req.body);
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
});

app.listen(3000, () => {
  console.log('MCP server listening on http://localhost:3000/mcp');
});
```

Each request creates a fresh server instance with a fresh rate limiter. This is the stateless tradeoff
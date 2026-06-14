## Ship It

Deploying an MCP client into a production GTM stack means treating it like any other piece of infrastructure: it needs health checks, retry logic, and a deployment pipeline. The zone table places this at Zone 13 — Production GTM Infrastructure. Your deploy pipeline ships your Clay tables, your n8n workflows, and any MCP-based enrichment services. SPF/DKIM/DMARC is your email infrastructure layer; MCP is your data integration layer.

The two production concerns that bite first are transport selection and failure recovery. Use `stdio` when your enrichment tool runs on the same machine as the client — a Lambda function spawning a local MCP server process, or an n8n worker with a bundled server. Use Streamable HTTP when the server is remote — a managed enrichment API exposed via MCP, or a shared server running in a different container. The `stdio` transport has no network failure modes but inherits process lifecycle risk (if the child process crashes, the session dies). The HTTP transport survives process restarts but introduces network latency, timeout, and retry semantics you must configure explicitly.

The code below wraps the client in a production-ready class with connection retry, tool caching, and structured error handling. It is the shape you would deploy inside a worker process that an n8n workflow or a Clay webhook calls.

```typescript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

class ProductionMcpClient {
  private client: Client | null = null;
  private toolCache: Map<string, { description: string; inputSchema: any }> = new Map();
  private readonly command: string;
  private readonly args: string[];
  private readonly maxRetries: number;

  constructor(command: string, args: string[], maxRetries = 3) {
    this.command = command;
    this.args = args;
    this.maxRetries = maxRetries;
  }

  async connect(): Promise<void> {
    let lastError: Error | null = null;
    for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
      try {
        const transport = new StdioClientTransport({
          command: this.command,
          args: this.args,
        });
        this.client = new Client(
          { name: "production-mcp-client", version: "1.0.0" },
          { capabilities: {} }
        );
        await this.client.connect(transport);
        await this.refreshToolCache();
        console.log(`Connected on attempt ${attempt}`);
        return;
      } catch (err) {
        lastError = err as Error;
        console.error(`Attempt ${attempt} failed: ${lastError.message}`);
        if (attempt < this.maxRetries) {
          await new Promise((r) => setTimeout(r, 1000 * attempt));
        }
      }
    }
    throw new Error(`Failed after ${this.maxRetries} attempts: ${lastError?.message}`);
  }

  private async refreshToolCache(): Promise<void> {
    if (!this.client) throw new Error("Not connected");
    this.toolCache.clear();
    const tools = await this.client.listTools();
    for (const tool of tools.tools) {
      this.toolCache.set(tool.name, {
        description: tool.description ?? "",
        inputSchema: tool.inputSchema,
      });
    }
    console.log(`Tool cache refreshed: ${this.toolCache.size} tools`);
  }

  async callTool(name: string, args: Record<string, unknown>): Promise<any> {
    if (!this.client) throw new Error("Not connected");
    if (!this.toolCache.has(name)) {
      await this.refreshToolCache();
      if (!this.toolCache.has(name)) {
        throw new Error(`Tool not found: ${name}`);
      }
    }
    const result = await this.client.callTool({ name, arguments: args });
    if (result.isError) {
      const errorText = result.content
        .filter((c: any) => c.type === "text")
        .map((c: any) => c.text)
        .join("\n");
      throw new Error(`Tool error: ${errorText}`);
    }
    return result.content;
  }

  async healthCheck(): Promise<boolean> {
    if (!this.client) return false;
    try {
      await this.client.ping();
      return true;
    } catch {
      return false;
    }
  }

  async close(): Promise<void> {
    if (this.client) {
      await this.client.close();
      this.client = null;
      this.toolCache.clear();
      console.log("Production client closed");
    }
  }
}

const prod = new ProductionMcpClient("npx", ["-y", "@modelcontextprotocol/server-everything"]);

await prod.connect();
console.log(`Healthy: ${await prod.healthCheck()}`);

const content = await prod.callTool("echo", { message: "production check" });
console.log(`Response: ${JSON.stringify(content)}`);

try {
  await prod.callTool("nonexistent_tool", {});
} catch (err) {
  console.log(`Expected error caught: ${(err as Error).message}`);
}

await prod.close();
```

The retry loop uses exponential backoff — 1s, 2s, 3s between attempts — because the most common failure mode for stdio transport is the child process not being ready when `connect()` fires. The tool cache avoids calling `tools/list` on every invocation; it refreshes only when a requested tool is not in the cache, which handles the `tools/list_changed` notification case implicitly. The `healthCheck` method uses the MCP `ping` primitive — a JSON-RPC request that expects a `pong` response — which is the protocol-level equivalent of a TCP health check.
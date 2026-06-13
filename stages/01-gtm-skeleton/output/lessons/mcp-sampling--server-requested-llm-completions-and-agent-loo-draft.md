# MCP Sampling — Server-Requested LLM Completions and Agent Loops

## Learning Objectives

1. Implement an MCP server that requests LLM completions from a connected client using the `sampling/createMessage` protocol method.
2. Configure sampling request parameters — `modelPreferences`, `maxTokens`, `system` — to control server-initiated completions.
3. Build an agent loop where a server iteratively calls the LLM to refine output until a convergence condition is met.
4. Compare MCP sampling vs. tool-calling as orchestration patterns and identify when each is appropriate.
5. Evaluate the permission and security model around server-requested completions, including human-in-the-loop approval gates.

---

## Beat 1: Hook — The Server Asks the LLM a Question

Normally the LLM calls your server. Sampling inverts this: your server calls the LLM. This is the primitive that turns an MCP tool from a passive function into an agent. A lead research server can read a profile, ask the LLM to score it, read the score, and ask again with different context — all without the user re-prompting. If you have ever wanted a Clay enrichment to "think harder" before returning a result, this is the mechanism that enables it.

---

## Beat 2: Concept — What Sampling Actually Is

Sampling is an MCP protocol capability (`sampling/createMessage`) where a **server** sends a request to the **client** asking it to run an LLM completion. The client — typically Claude Desktop, an IDE, or a custom host — decides whether to execute it, possibly after user approval. The server specifies messages, system prompt, model preferences, and token limits. The client returns the LLM's response. This creates a bidirectional channel: the user prompts the LLM, the LLM calls the server, the server calls the LLM back. That loop is the foundation of agentic behavior in MCP.

Key distinctions:
- **Tool calling**: client → LLM → server (server is passive)
- **Sampling**: client → LLM → server → LLM (server is active)
- The server never has API keys. The client holds them. The server just *asks*.

---

## Beat 3: Mechanism — The Protocol, Parameters, and Loop Pattern

**Protocol flow:**

1. Server detects a condition during tool execution (e.g., result is ambiguous, needs summarization, requires multi-step reasoning).
2. Server constructs a `CreateMessageRequest` with:
   - `messages`: array of role/content pairs (the conversation context the server wants the LLM to see)
   - `system`: optional system prompt for this specific completion
   - `maxTokens`: hard ceiling on response length
   - `modelPreferences`: hints for model selection (`hints[].name`, `costPriority`, `speedPriority`, `intelligencePriority`)
   - `temperature`: sampling randomness
3. Server calls `this.client.request(...)` (in TypeScript SDK) or equivalent.
4. Client receives the request. Client may:
   - Auto-approve based on capability negotiation
   - Prompt the user for approval
   - Reject the request
5. Client runs the completion against its configured LLM provider.
6. Client returns `CreateMessageResult` containing `role`, `content`, and `model`.
7. Server inspects the result and decides: return to user, or loop again with new context.

**Agent loop pattern:**

```
tool execution begins
  → server gathers data
  → server asks LLM: "analyze this"
  → LLM responds
  → server checks: is response sufficient?
    → no: server asks LLM again with refined context
    → yes: server returns final result to user
```

Convergence conditions prevent infinite loops: max iterations, confidence thresholds, or explicit "DONE" signals in the LLM's response.

**Capability negotiation:** During initialization, the client advertises `capabilities.sampling` if it supports this. The server must check for this capability before sending sampling requests — not all clients support it.

---

## Beat 4: Use It — GTM Application: Iterative Enrichment Loops

[CITATION NEEDED — concept: MCP sampling for GTM enrichment loops]

This maps to **Zone 3: Enrichment** in the GTM topic map. The specific mechanism: an MCP server performing lead or account research can use sampling to iteratively refine enrichment results rather than returning raw data on the first pass.

**Concrete pattern — Research-then-Score Loop:**

A server tool like `research_account(domain)` could:
1. Fetch firmographic data from an API.
2. Ask the LLM: "Given this firmographic data, identify the top 3 buying signals."
3. Parse the LLM response. If signals are weak, fetch additional data (news, funding, hires).
4. Ask the LLM again with enriched context: "Now with this additional data, re-evaluate buying signals."
5. Return the refined assessment to the Clay waterfall as a scored output.

This is the mechanism behind what GTM teams call "AI enrichment" — it is not the LLM making a single pass over data, it is a loop where the tool decides whether its own output is good enough and refines it. Without sampling, every enrichment is one-shot. With sampling, enrichment is iterative.

**Exercise hooks:**
- **Easy**: Configure a server with sampling capability and make a single `createMessage` request that returns a structured enrichment result.
- **Medium**: Build a two-step enrichment loop: fetch data → ask LLM to score → return score. No iteration, just the bidirectional call.
- **Hard**: Implement a convergence loop where the server iterates up to 3 times, checking if the LLM's confidence score exceeds a threshold before returning.

---

## Beat 5: Ship It — Build a Working Sampling Server with an Agent Loop

Build a complete MCP server that exposes a tool requiring iterative LLM reasoning. The server will:

1. Declare `sampling` capability requirements during initialization.
2. Expose a tool (e.g., `analyze_icp_fit`) that receives company data.
3. Inside the tool handler, construct a `CreateMessageRequest` with a system prompt defining ICP criteria.
4. Parse the LLM response for a confidence score.
5. If below threshold, append the LLM's reasoning as context and request again.
6. Return the final scored result to the user.

**Code example** (TypeScript MCP SDK, runs in Node.js, observable output via console.log):

```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

const server = new Server(
  { name: "icp-scorer", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "score_icp_fit",
      description: "Scores a company against ICP criteria using iterative LLM reasoning",
      inputSchema: {
        type: "object",
        properties: {
          company_name: { type: "string" },
          industry: { type: "string" },
          employee_count: { type: "number" },
          funding_stage: { type: "string" },
        },
        required: ["company_name", "industry", "employee_count", "funding_stage"],
      },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name !== "score_icp_fit") {
    return { content: [{ type: "text", text: "Unknown tool" }], isError: true };
  }

  const args = request.params.arguments as {
    company_name: string;
    industry: string;
    employee_count: number;
    funding_stage: string;
  };

  const MAX_ITERATIONS = 3;
  const CONFIDENCE_THRESHOLD = 0.8;
  let iteration = 0;
  let conversationMessages = [
    {
      role: "user" as const,
      content: {
        type: "text" as const,
        text: `Company: ${args.company_name}\nIndustry: ${args.industry}\nEmployees: ${args.employee_count}\nFunding: ${args.funding_stage}\n\nScore this company against an ICP for a B2B SaaS selling to growth-stage fintech companies (50-500 employees, Series A-C). Return a JSON object with "score" (0-1), "confidence" (0-1), and "reasoning" (string).`,
      },
    },
  ];

  while (iteration < MAX_ITERATIONS) {
    iteration++;
    console.error(`[icp-scorer] Sampling iteration ${iteration}/${MAX_ITERATIONS}`);

    const samplingRequest = {
      method: "sampling/createMessage" as const,
      params: {
        messages: conversationMessages,
        system: "You are an ICP scoring analyst. Always respond with valid JSON containing score, confidence, and reasoning keys. If you lack information, set confidence low.",
        maxTokens: 500,
        temperature: 0.3,
      },
    };

    let response;
    try {
      response = await server.server.request(samplingRequest, {
        method: "sampling/createMessage",
      });
    } catch (err) {
      console.error("[icp-scorer] Sampling request failed:", err);
      return {
        content: [
          {
            type: "text" as const,
            text: `Sampling failed after ${iteration} iterations. Client may not support sampling capability.`,
          },
        ],
        isError: true,
      };
    }

    const responseText = response.content.type === "text" ? response.content.text : "";
    console.error(`[icp-scorer] LLM response (iteration ${iteration}): ${responseText.substring(0, 200)}...`);

    let parsed;
    try {
      const jsonMatch = responseText.match(/\{[\s\S]*\}/);
      if (!jsonMatch) throw new Error("No JSON found");
      parsed = JSON.parse(jsonMatch[0]);
    } catch {
      console.error("[icp-scorer] Failed to parse LLM response as JSON, retrying...");
      conversationMessages.push({ role: "assistant" as const, content: response.content });
      conversationMessages.push({
        role: "user" as const,
        content: {
          type: "text" as const,
          text: "Your previous response was not valid JSON. Please respond with ONLY a JSON object containing score, confidence, and reasoning.",
        },
      });
      continue;
    }

    console.error(`[icp-scorer] Parsed — score: ${parsed.score}, confidence: ${parsed.confidence}`);

    if (parsed.confidence >= CONFIDENCE_THRESHOLD || iteration === MAX_ITERATIONS) {
      const finalResult = {
        company: args.company_name,
        icp_score: parsed.score,
        confidence: parsed.confidence,
        reasoning: parsed.reasoning,
        iterations_used: iteration,
        converged: parsed.confidence >= CONFIDENCE_THRESHOLD,
      };
      console.error(`[icp-scorer] Final result: ${JSON.stringify(finalResult)}`);
      return {
        content: [{ type: "text" as const, text: JSON.stringify(finalResult, null, 2) }],
      };
    }

    conversationMessages.push({ role: "assistant" as const, content: response.content });
    conversationMessages.push({
      role: "user" as const,
      content: {
        type: "text" as const,
        text: `Your confidence was ${parsed.confidence}, below the ${CONFIDENCE_THRESHOLD} threshold. The score was ${parsed.score}. Re-evaluate with more specificity. What additional signals would increase confidence? Provide your best assessment.`,
      },
    });
  }

  return {
    content: [{ type: "text" as const, text: "Max iterations reached without convergence" }],
    isError: true,
  };
});

const transport = new StdioServerTransport();
await server.connect(transport);
console.error("[icp-scorer] Server running on stdio");
```

**Testing this**: Connect via Claude Desktop or a custom MCP client that supports `sampling`. Call `score_icp_fit` with sample company data. The server will log each iteration to stderr, showing the loop in action.

**Exercise hooks:**
- **Easy**: Run the server as-is. Connect from a sampling-capable client. Observe the iteration logs in stderr.
- **Medium**: Modify the convergence condition to also check if the score itself changes less than 0.05 between iterations (stability check).
- **Hard**: Add a second tool that fetches real data (e.g., from a mock API), feeds it into the sampling loop as additional context on iteration 2+, and returns a final verdict combining raw data with LLM reasoning.

---

## Beat 6: Evaluate — Assessment

**Conceptual questions:**

1. In the MCP sampling flow, who holds the LLM API key — the server or the client? What is the security implication of this design choice?

2. A server sends a `CreateMessageRequest` with `modelPreferences: { hints: [{ name: "claude-3-5-sonnet" }], costPriority: 0.9 }`. What does `costPriority: 0.9` signal to the client, and when would you set it high vs. low?

3. You are building an enrichment tool that makes 3 sampling requests per invocation. Each request waits for the previous response. A user calls this tool 50 times in a batch. Describe the bottleneck and propose an architectural alternative.

4. Compare these two patterns for building an agent loop: (a) MCP sampling where the server iteratively calls the LLM, (b) client-side orchestration where the LLM calls the server tool in a loop. When is (a) better? When is (b) better?

5. A sampling request returns `isError: true` from the client. List three possible causes and the debugging steps for each.

**Mechanism identification:**

6. Given a transcript of MCP protocol messages, identify which steps are tool calls (client→server) and which are sampling requests (server→client). Draw the message flow.

7. A server's sampling loop runs 10 iterations without converging. The `maxTokens` is set to 100. Diagnose the likely cause and fix it.

**Implementation check:**

8. Write the capability check a server should perform before sending its first `createMessage` request. What happens if you skip this check against a non-sampling client?

---

## GTM Redirect Summary

**Primary cluster**: Zone 3 — Enrichment. The iterative LLM reasoning pattern enabled by MCP sampling is the mechanism behind "AI enrichment" in GTM workflows. When an enrichment step in a Clay waterfall needs to reason over data rather than just transform it, this is the underlying protocol primitive.

**Secondary cluster**: Zone 6 — Infrastructure. Sampling is a protocol-level capability that affects how GTM tools are architected. Understanding it is necessary for building custom MCP integrations that go beyond simple data fetch-and-return patterns.

**Foundational note**: If the specific GTM application is unclear, the redirect is: "MCP sampling is foundational for building agentic enrichment tools in Zone 3. The mechanism (server-initiated LLM loops) is what makes iterative scoring, research, and personalization possible without custom client orchestration."
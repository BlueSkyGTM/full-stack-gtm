# The Agent Loop: Observe, Think, Act

## Learning Objectives

1. Implement a functional observe-think-act loop that terminates on a condition
2. Compare single-shot prompting against agent-loop execution using concrete token logs
3. Diagnose infinite-loop failures in agent systems from trace output
4. Configure tool definitions that an agent loop can discover and invoke correctly
5. Trace state mutation across iterations of an agent loop and predict final state

---

## Beat 1: Hook — What Breaks Without a Loop

A single LLM call is stateless: it receives a prompt, returns a completion, and forgets everything. That works for drafting an email. It fails the moment you need the model to check whether the email actually sent, retry on failure, or adapt based on new information. The agent loop is the pattern that turns a completion endpoint into a system that can pursue goals over time. Without it, every "agentic" tool is just a fancy prompt chain with a marketing budget.

---

## Beat 2: Concept — The Three-Phase Cycle

The loop has three phases that repeat until a termination condition is met:

- **Observe**: Ingest new information — user message, tool output, environment state. This is the only way the loop learns anything changed.
- **Think**: The model reasons over the accumulated observation history and decides what to do next. This is a standard LLM completion call, but the prompt includes the full observation trail.
- **Act**: Execute a tool, return a message, or modify state. The action produces a new observation, which feeds the next iteration.

Termination happens when the model emits a "done" signal (a special token, a stop phrase, or a tool called `finish`) or when a hard iteration cap is hit.

Key distinction: the loop is not recursion. Each iteration is a discrete state transition. The model does not "remember" — the context window carries compressed history forward, and every iteration reprocesses it.

---

## Beat 3: Mechanism — State, Tools, and the Halting Problem

The mechanism has four components:

1. **State store**: A list (or similar structure) that accumulates observations and actions. Each iteration appends to it. The entire list is sent to the model on every `Think` call.
2. **Tool registry**: A mapping of tool names to executable functions. The model returns structured tool-call requests; the loop dispatches them to the registry and collects results.
3. **Decision parser**: Logic that inspects the model's output to determine whether to invoke a tool, return a final answer, or error out.
4. **Iteration guard**: A counter that forces termination after N steps. This is not optional — without it, a model stuck in a reasoning loop will run until you run out of tokens or money.

The halting problem applies: you cannot guarantee the agent will finish. The iteration cap is the pragmatic answer. Most production agents cap at 5–15 iterations depending on task complexity.

Context window pressure: each iteration adds tokens to the state. A 20-iteration loop with verbose tool outputs can exhaust a 128k context window. Production systems implement summarization or truncation strategies — [CITATION NEEDED — concept: context window management in multi-turn agent loops].

---

## Beat 4: Code — Running the Loop

Build a minimal agent loop that searches for a target number using a "hotter/colder" tool. The loop observes tool output, thinks by calling the model, acts by guessing, and terminates when the guess is correct or the cap is reached.

Exercise hooks:
- **Easy**: Modify the termination condition to also stop if the model repeats the same guess twice in a row.
- **Medium**: Add a second tool (`hint_provider`) that returns "above" or "below" and observe how the loop's iteration count changes.
- **Hard**: Implement a token budget that forces termination when cumulative input tokens exceed a threshold, and log the token cost per iteration.

---

## Beat 5: Use It — Agent Loops in GTM Workflows

The Clay waterfall enrichment pattern is an agent loop variant. Observe: check if a field (e.g., company headcount) is populated. Think: if empty, select the next enrichment provider. Act: call the provider's API. The loop repeats across providers until the field is filled or providers are exhausted — which is the iteration cap.

Specific GTM applications:

- **Lead research agents**: Observe (read LinkedIn profile), Think (determine if the lead fits ICP), Act (enrich with company data or disqualify). Loop until the agent has enough signal to score.
- **Outreach sequence agents**: Observe (check if prospect opened email), Think (decide next touchpoint), Act (send follow-up or mark as unresponsive). Loop over the sequence cadence.
- **Data enrichment pipelines**: The waterfall pattern — iterate through data providers until a field is filled. This is structurally identical to the observe-think-act loop, where "Think" is a routing rule rather than an LLM call.

GTM cluster: **Zone 03 — Enrichment & Data Quality** (waterfall pattern) and **Zone 07 — AI Agents & Workflows** (agent orchestration).

---

## Beat 6: Ship It — Production Concerns

The loop works in a notebook. Production introduces three problems the lesson doesn't fully solve:

1. **Latency accumulation**: Each iteration adds an LLM inference round-trip (200ms–2s). A 10-iteration loop with tool calls can take 15–30 seconds. Users will not wait. Production systems stream intermediate observations or run loops asynchronously with webhook callbacks.
2. **Cost per iteration**: Every re-sends the full state. A 5-iteration loop with a 4k-token state costs ~20k input tokens. At scale, this compounds. [CITATION NEEDED — concept: token cost optimization patterns for agent loops in production].
3. **Observability**: When an agent loop goes wrong, you need the full trace — every observation, every model decision, every tool result. Ship a logging layer that records each iteration's inputs and outputs before deploying any agent loop to production.

Assignment: Instrument the Beat 4 loop with iteration-level logging (iteration number, tokens consumed, action taken). Run it 20 times against the same target and compute the average iteration count and token cost. Report both numbers.
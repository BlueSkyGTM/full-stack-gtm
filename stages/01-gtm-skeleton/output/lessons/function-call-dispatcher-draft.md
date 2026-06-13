# Function Call Dispatcher

## Learning Objectives

1. Build a name-to-handler registry that routes LLM tool calls to executable Python functions
2. Implement the agent loop: LLM response → parse tool call → dispatch → inject result → repeat until done
3. Trace a full dispatch cycle and identify where it breaks (missing handler, bad args, infinite loop)
4. Compare a hand-rolled dispatcher to OpenAI's tool-calling API surface
5. Configure permission boundaries that restrict which tools an agent can invoke

---

## Beat 1: Concept

The function call dispatcher is the switchboard between an LLM's *decision* to use a tool and the *execution* of that tool. When an LLM returns a structured request to call "lookup_company(domain='acme.com')", something has to find the right function, pass the arguments, catch the return value, and hand it back. That something is the dispatcher.

---

## Beat 2: Mechanism

Three components, one loop. **Registry**: a dict mapping function names to callable handlers. **Dispatch function**: looks up the name in the registry, calls the handler with the LLM-provided args, catches errors. **Agent loop**: send messages to LLM → check if response contains tool calls → dispatch each one → append results as tool messages → send again. The loop terminates when the LLM returns a plain text response instead of a tool call. Failure modes: name not in registry (KeyError), handler raises an exception, LLM keeps calling tools forever (max iterations guard).

---

## Beat 3: Implement

Build the three components from scratch in ~60 lines of Python against the OpenAI API. First, define two tool functions (`lookup_company`, `score_lead`) and register them in a dict. Then write `dispatch(tool_call)` that extracts name + args, calls the handler, returns the result. Then write `agent_loop(messages, max_turns=5)` that runs the cycle. Print the full conversation at each step so the dispatch trace is observable. No frameworks — just `openai` and `json`.

**Exercise hooks:**
- *(Easy)* Add a third tool to the registry and observe the LLM choose it based on the prompt
- *(Medium)* Modify the dispatcher to log every call (name, args, latency, result) to a list, then print the trace after the loop completes
- *(Hard)* Implement a permission system: each tool has a `required_role` field; the dispatcher rejects calls when the session role doesn't match. Test that a "viewer" session cannot call `score_lead`.

---

## Beat 4: Use It

**GTM Cluster 09 — Agents, tool use, function calling.** The function call dispatcher *is* the mechanism inside every workflow automation tool. An n8n workflow node is a registered function. The wires between nodes are the agent loop deciding what fires next. In cold calling infrastructure (Zone 2.2), the "task router" is a dispatcher: it receives the LLM's decision, looks up the right tool (enrich lead → draft script → log outcome), and executes. You built the algorithm here; wiring it into n8n/Make later is the same pattern with a GUI on top.

---

## Beat 5: Ship It

Production dispatchers need: timeout enforcement per tool (a slow HTTP call shouldn't stall the loop), error formatting that the LLM can recover from (return "Error: rate limited, retry in 60s" as a tool message, don't crash the loop), and a max-turns guard to prevent infinite dispatch cycles. Logging every dispatch decision (name, args, wall time, success/fail) is non-negotiable — this is your audit trail when the agent does something wrong. Permission boundaries go here too: the dispatcher is the enforcement layer, not the LLM.

---

## Beat 6: Evaluate

**Quiz hooks** (testable against objectives):
- Given a registry `{"a": fn_a, "b": fn_b}` and an LLM response requesting tool `"c"`, what does the dispatcher do and what message does it return to the LLM?
- Trace a 2-turn agent loop: show the message array state after each dispatch cycle (assistant message with tool call → tool result message → assistant message).
- A dispatcher logs 47 turns on a single prompt. What guard failed, and how do you fix it?
- Two tools have overlapping capability (`enrich_lead` and `enrich_company`). The LLM calls the wrong one. Is this a dispatcher bug or a tool definition bug? Explain.

**Exercise hooks:**
- *(Easy)* Run the provided dispatcher code with a prompt that triggers exactly 2 tool calls. Print the message array length at each step.
- *(Medium)* Add a `max_turns=3` guard. Write a prompt that would cause an infinite loop without it. Confirm the guard stops execution and prints a warning.
- *(Hard)* Instrument the dispatcher with timing. Run the same prompt 5 times, collect latency per tool call, and print a summary table. Identify which tool is the bottleneck.
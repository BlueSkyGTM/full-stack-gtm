# The Minimal Agent Workbench

## Hook

You've used agents that feel like magic—until they loop forever, call the wrong tool, or hallucinate a function that doesn't exist. Here's what's actually happening under the hood, and why most "agent platforms" are just a while-loop with lipstick.

## Concept

The agent loop is observe→plan→act→observe. A minimal workbench implements this as a controlled cycle: a prompt, a list of callable tools with typed schemas, and a stopping condition. Introduce the ReAct pattern (Reason + Act) as the core mechanism, then the tool-calling protocol that binds LLM output to executable functions. No framework yet—just the loop, the tool schema, and the dispatch table.

## Demo

Build a 60-line agent loop in Python that takes a user goal, a list of tool functions (with JSON-serializable signatures), calls Claude via the API, parses tool_use blocks, dispatches to real Python functions, and feeds results back until the model returns a stop or max_iterations is hit. Every step prints to stdout. [CITATION NEEDED — concept: minimal agent loop reference implementation pattern]

## Debrief

Walk through what happened in the demo: where the model decided to act vs. answer, how tool output changed the next reasoning step, where it could have spiraled (and didn't, because of max_iterations). Map each line to the ReAct cycle. Name the failure modes: infinite loops, tool schema drift, ambiguous stop conditions.

## Use It

This is the same loop that powers GTM enrichment waterfalls—prospect → look up company → check intent data → score → decide next action. The agent loop replaces hardcoded Zapier chains with conditional multi-step reasoning. [CITATION NEEDED — concept: GTM enrichment waterfall as agent loop] Exercise hook: wire a two-tool agent that looks up a domain and classifies ICP fit (easy), or add a third tool that branches on ICP result (medium), or add retry logic with a reflection step on tool failure (hard).

## Ship It

Build a minimal agent workbench that solves a real multi-step task: given a company name, research the domain, classify the industry, and write a one-line positioning hook. Must use at least two tools, must print each step, must stop within 6 iterations. Include a failure log showing what happened when it didn't work. [CITATION NEEDED — concept: agent evaluation criteria for GTM use cases]
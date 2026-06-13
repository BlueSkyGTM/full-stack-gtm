# Capstone 01 — Terminal-Native Coding Agent

## Hook
You've used Claude Code. Now you'll build a stripped-down version: a terminal agent that reads files, executes commands, and iterates on its own output using a tool-use loop. This is the capstone that proves you can wire an LLM to a real environment.

## Learn It
The ReAct loop: reason → act → observe → repeat. Trace how an agent decomposes a task, selects a tool, parses output, and decides whether to continue or terminate. Cover tool schemas, the agent state machine, and why most agent failures are parsing failures, not reasoning failures.

## Use It
GTM redirect: this agent architecture maps directly to Zone 1 enrichment workflows — an agent that reads a company's website, extracts ICP signals, and writes structured output. The same loop that reads a file and runs a linter can be repurposed to scrape a domain and populate a CRM field. [CITATION NEEDED — concept: terminal agent applied to Zone 1 enrichment workflows]

## Build It
Three exercises of increasing scope:
- **Easy**: Wire a single tool-call round trip — send a prompt with one tool definition, capture the model's tool invocation, execute it locally, and print the result.
- **Medium**: Implement the full ReAct loop with three tools (read file, list directory, run shell command). Agent runs until it produces a final text answer or hits a step limit.
- **Hard**: Add error recovery — if a tool call returns a non-zero exit code or malformed JSON, feed the error back into the loop and let the agent self-correct. Log every turn to a JSONL file for post-run inspection.

## Ship It
Package the agent as a single Python script with a `requirements.txt`. Agent accepts a task string as a CLI argument, executes autonomously, and writes results to stdout. Include the JSONL trace file as a sibling output. GTM redirect: this script structure is the same skeleton you'd deploy as a Clay webhook endpoint for automated enrichment — swap the CLI arg for an HTTP payload and you have a production Zone 1 worker.

## Push It
Extended challenges for independent work:
- Add a `--dry-run` flag that prints what the agent *would* do without executing shell commands. Evaluate: does the agent plan correctly when it can't observe intermediate output?
- Replace the hardcoded step limit with a token-budget limiter. Measure how different task phrasings burn through the budget.
- Implement a human-in-the-gate checkpoint: before any shell command execution, pause and require `y/n` confirmation. Log which commands the human rejects and analyze patterns.
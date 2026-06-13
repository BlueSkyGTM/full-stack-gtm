# Capstone 16 — GitHub Issue-to-PR Autonomous Agent

## Hook

You've built agents that call tools, agents that plan, and agents that iterate. Now wire all three into a single loop that reads a GitHub Issue, writes code to solve it, opens a Pull Request, and stops. This is the integration test for everything in the preceding fifteen lessons.

## Concept

The issue-to-PR pipeline decomposes into four stages — **fetch**, **plan**, **edit**, **propose** — each with its own failure mode. The mechanism is a state machine: the agent holds a representation of the repo's file tree, the issue's acceptance criteria, and the diff so far. Each loop iteration asks the LLM to classify the current state (need more context / ready to edit / edit complete / ready to PR) and act accordingly. The key design tension is **stopping criteria**: the agent must decide when the diff satisfies the issue without human confirmation, which makes this a concrete instance of the "when to stop" problem from the planning lessons.

## Build

**Easy:** Write the fetch stage — call the GitHub API, extract issue title, body, labels, and comments into a structured object. Print the parsed result to confirm the shape.

**Medium:** Add the plan stage — send the issue object plus the file tree to the LLM with a system prompt that forces it to output a JSON plan (list of files to read, list of files to edit, expected test command). Print the plan.

**Hard:** Complete the loop — after the plan, iterate: read files, ask the LLM to produce a unified diff, apply the diff with a dry-run flag, run tests, and if tests pass open the PR via the GitHub API. Print the PR URL.

## Use It

This is an internal-developer-productivity automation. The closest GTM cluster is **Zone 4 — Workflow Automation**: the same state-machine-over-LLM-calls pattern powers automated enrichment pipelines where an agent reads a CRM alert, fetches company data, writes a research brief, and posts it to a Slack channel. The mechanism is identical — fetch → plan → edit → propose — only the "edit" step produces prose instead of code. [CITATION NEEDED — concept: Zone 4 workflow automation agent pattern]

## Ship It

Exercise hook: deploy the agent as a GitHub Action triggered by the `issue_comment` event with the substring `/fix`. Add a timeout of three LLM calls and a circuit breaker that comments on the issue if the agent fails. Print the Action run URL to confirm deployment.

## Evaluate

Exercise hook: given a recording of three agent runs (one success, one timeout, one test failure), write a short post-mortem identifying which stage each run failed at and whether the failure was recoverable with one more loop iteration. Output the post-mortem as a structured list of (run_id, failure_stage, recoverable: bool, rationale).
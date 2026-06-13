# Multimodal Agents and Computer-Use (Capstone)

## Hook

A multimodal agent doesn't just call APIs—it sees a screen, decides what to click, and types into fields that were never designed for automation. This beat opens with a live trace of an agent navigating a real web interface, failing, retrying, and eventually completing a multi-step task, establishing why perception-plus-action is fundamentally different from tool-calling alone.

## Concept

Explains the observe-reason-act loop as it applies to computer-use: screenshot capture (or accessibility tree extraction) as the observation space, coordinate-based or element-based action primitives as the action space, and the role of grounding (mapping visual pixels to semantic actions). Covers two architectural patterns: (1) screenshot + coordinate prediction (e.g., Anthropic computer use, OmniParser) and (2) DOM/accessibility-tree + structured tool calls (e.g., browser-use, Playwright-based agents). Compares latency, reliability, and failure modes of each. Introduces the concept of action space design—why "click at (x,y)" and "click element by selector" have different error profiles.

## Demo

A working Python script that runs a headless browser task using an agent loop: the agent receives a screenshot, issues a coordinate-based or element-based action, observes the result, and iterates until task completion. Output is a step-by-step trace showing each observation, the agent's reasoning, the action taken, and the final result. Uses either Anthropic's computer-use beta API or an open-source browser agent framework, with clear fallback instructions. No mock screenshots—real browser interaction or a clearly documented simulation mode.

## Use It

**GTM Redirect: Zone 3 (Enrichment & Research) and Zone 4 (Outreach Execution).** Computer-use agents fill the gap where no API exists: scraping data from legacy CRMs, navigating LinkedIn Sales Navigator when API access is restricted, completing multi-step research across company websites that lack structured endpoints. The mechanism is an agent loop with visual grounding; the GTM application is automating research and data-entry workflows that previously required a human to click through interfaces. Includes one concrete example: an agent that visits a company's careers page, reads open roles, and writes structured enrichment data. [CITATION NEEDED — concept: GTM enrichment workflows requiring GUI automation vs. API access]

**Exercise hooks:**
- *Easy:* Modify the demo agent to extract a specific data point from a target page and print it as structured JSON.
- *Medium:* Build an agent loop that navigates a multi-page form (e.g., a mock lead-capture flow), fills fields from a provided data object, and confirms submission.
- *Hard:* Implement a retry-and-recovery mechanism: when the agent encounters an unexpected state (modal, CAPTCHA placeholder, navigation error), it must detect the anomaly, describe it, and attempt a recovery path before failing.

## Ship It

Covers the production concerns specific to computer-use agents: non-determinism and retry budgets (the same task may take 3 steps or 12), screenshot resolution and scaling across display sizes, session management and state cleanup, credential handling when the agent must log into a service, and cost modeling (each observe-reason-act cycle is a model call). Includes a deployment pattern: wrap the agent loop in a task queue with a maximum-step limit, record full traces for debugging, and expose a health-check endpoint. Provides a working scaffold for a containerized agent worker that pulls tasks from a queue, runs the observe-reason-act loop, and writes results to a file or database. [CITATION NEEDED — concept: production deployment patterns for GUI-automation agents in enterprise environments]

## Assess

Quiz questions grounded in observable behavior from the demo and exercises: identifying which architectural pattern (screenshot-based vs. DOM-based) is in use given a trace, predicting failure modes when the action space is misconfigured, explaining why coordinate-based actions fail on responsive layouts, evaluating a trace to determine at which step the agent's observation was insufficient and what additional context would have prevented the error. No trivia—every question references the mechanisms and outputs from this lesson.
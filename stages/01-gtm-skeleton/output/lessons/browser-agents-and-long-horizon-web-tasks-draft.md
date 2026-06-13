# Browser Agents and Long-Horizon Web Tasks

## Beat 1: Hook

Why "just call the API" fails for half the web. Many data sources and GTM workflows live behind JavaScript-rendered pages, login walls, and multi-step form flows that no API covers. This lesson addresses the mechanism gap between static HTTP requests and autonomous web interaction.

## Beat 2: Concept

**Mechanism first.** A browser agent is a loop: observe DOM state → select an action → execute → observe new state → repeat until done or stuck. The hard problems are action space definition (what can the agent do), state representation (what does it see), and recovery from failure. Long-horizon means the task requires dozens or hundreds of these loops, which introduces compounding error and context window limits.

Key concepts to cover:
- **Action primitives**: click, type, scroll, navigate, extract — and how they compose into higher-level actions
- **DOM-as-observation**: accessibility trees vs. screenshots vs. raw HTML as agent inputs
- **Planning decomposition**: breaking "research this company" into a sequence of sub-goals
- **Memory and context management**: what the agent remembers across steps when the task spans 50+ actions
- **Recovery patterns**: retry, backtrack, re-plan when an action fails or produces unexpected state

**Tools named after mechanism:** Browser Use implements the observe-act loop with an LLM selecting actions against a simplified DOM. Playwright is the browser automation layer that executes the primitives. [CITATION NEEDED — concept: specific browser agent benchmarks and their action space definitions]

## Beat 3: Demo

Build a minimal browser agent loop in Python that demonstrates the observe-act architecture without requiring a live browser. Uses a simulated DOM state machine to show how action selection, state observation, and termination detection work. All code runs in terminal with observable output.

**Exercise hooks:**
- Easy: Modify the action selector to handle one additional action primitive and print the resulting state transitions
- Medium: Add a retry mechanism that detects when an action produces no state change and re-plans
- Hard: Implement a token-budget tracker that forces the agent to summarize and compress its action history when context exceeds a threshold

## Beat 4: Use It

**GTM Redirect:** This maps directly to Zone 1 (Research & Enrichment) — specifically the Clay waterfall pattern where enrichment cascades through multiple data sources. Browser agents fill the gap where no API exists: scraping LinkedIn profiles, extracting data from company websites with heavy JS rendering, and navigating multi-page government or regulatory databases. [CITATION NEEDED — concept: specific Clay waterfall enrichment steps that use browser automation vs. API calls]

Also maps to Zone 2 (Outreach) for workflows like: log into a sales tool, navigate to a contact record, log an activity, move a deal stage — tasks that require multi-step interaction with a web UI that has no API endpoint.

**Exercise hooks:**
- Easy: Design a 5-step action sequence for extracting company headcount from a JS-rendered careers page
- Medium: Write a state validator that checks whether a browser agent has actually landed on the correct LinkedIn profile vs. a login wall or redirect
- Hard: Implement a cost estimator that calculates LLM token spend for a 100-step web research task and compares it to the cost of a human doing the same task at $30/hr

## Beat 5: Ship It

Production concerns for browser agents in GTM pipelines:
- **Session management**: handling cookies, auth tokens, and session expiry mid-task
- **Rate limiting and detection**: behavioral fingerprinting, request throttling, rotating user agents
- **Reliability patterns**: checkpointing state so a failed run can resume instead of restart
- **Observability**: logging every action, screenshot, and DOM snapshot for debugging failed runs
- **Cost control**: token budgets per task, early termination when cost exceeds expected value

**Exercise hooks:**
- Easy: Write a checkpoint serializer that saves the agent's current URL, form state, and action count to JSON after every 10 actions
- Medium: Implement a detection circuit breaker that stops the agent when it encounters a CAPTCHA or login wall three times in a row
- Hard: Build a cost-aware planner that estimates total token cost before starting a task and aborts if the estimate exceeds a configurable threshold

## Beat 6: Evaluate

Assessment questions grounded in the mechanisms above. No trivia — every question tests whether the practitioner can reason about trade-offs in real browser agent deployments.

**Sample assessment hooks:**
- Compare accessibility-tree-based observation vs. screenshot-based observation for a form-filling task. Which fails first and why?
- Given a 200-step task with a 2% per-step failure rate, calculate the probability of completing the task without any recovery mechanism. Then calculate it with a retry-once policy.
- A browser agent is extracting data from 50 company websites. 15 of them have cookie consent banners that block the page. Describe the observation-action sequence to detect and dismiss the banner generically vs. site-specifically. What are the engineering tradeoffs?
- Design a test harness that can verify a browser agent's behavior without running against a live website.

---

**Learning Objectives (for docs/en.md):**
1. Implement an observe-act loop that selects DOM actions based on state observations
2. Compare accessibility-tree, screenshot, and raw-HTML observation mechanisms and their failure modes
3. Detect and recover from failed actions in a multi-step browser task
4. Estimate token cost and failure probability for long-horizon web tasks
5. Configure checkpointing and circuit-breaker patterns for production browser agent deployments
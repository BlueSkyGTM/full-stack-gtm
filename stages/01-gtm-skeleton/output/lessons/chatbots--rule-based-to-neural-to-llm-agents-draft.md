# Chatbots — Rule-Based to Neural to LLM Agents

## Hook It
The "chatbot" label masks three entirely different architectures. A rule-based bot is a state machine. A neural chatbot is a pattern-to-pattern mapper. An LLM agent is a reasoning loop with tool access. Calling all three "chatbots" is the source of most failed deployments. This lesson disambiguates the paradigms so you can select the right one for the right job.

## Ground It
Prerequisites: basic Python, HTTP request/response cycle, simple state machines. The lesson traces chatbot evolution through three architectural phases, each with different failure modes, cost profiles, and control surfaces. No ML background required — the neural section covers what you need inline.

## Explain It
Three paradigms, each explained by mechanism before naming tools:

**Rule-Based**: Finite state machine over user inputs. Pattern matching (regex/keyword) routes to canned responses. ELIZA demonstrated this in 1966 — keyword detection + template filling. Mechanism: `if input contains X, respond with Y`. Maintainability collapses around 50-100 rules. State tracking is manual and brittle.

**Neural (Intent-Based)**: Supervised classifier maps user utterance → intent class. A separate entity extractor pulls slot values. Dialogue management (rule-based or learned) decides next action. Mechanism: embedding lookup → classification head → intent label → response template. Rasa and Dialogflow implement this. Requires labeled training data. Fails on out-of-domain inputs silently.

**LLM Agents**: Autoregressive language model generates responses token-by-token. Tool/function calling extends this: the model outputs structured requests to external APIs, receives results, and continues reasoning. The ReAct pattern (Reason + Act) loops: observe → think → act → observe. Mechanism: prompt context → token prediction → parse action → execute tool → append result → repeat. No labeled data required, but cost per interaction is 100-1000x higher than rule-based.

## Use It
GTM redirect: Zone 4 (Engage) — specifically conversational qualification and intelligent routing. Rule-based bots handle linear qualification flows (budget? timeline? authority?). Neural/intent bots handle multi-turn FAQ at scale. LLM agents handle open-ended buyer conversations where the surface area of possible questions exceeds any rule set.

[CITATION NEEDED — concept: Clay AI formula columns for generating personalized outreach messages based on enrichment data]

The mechanism-to-tool mapping: Clay's AI enrichment fields implement an LLM call over structured company/contact data — this is the "agent with one tool" pattern where the tool is Clay's own data graph.

## Build It
Three exercises, one per paradigm, all terminal-runnable:

**Easy — Rule-Based**: Build a 5-state qualification bot using a dictionary-based state machine. Observable output: full conversation trace printed to terminal showing state transitions.

**Medium — Intent Classification**: Train a simple intent classifier on a toy dataset (10 intents, 5 examples each) using scikit-learn. Feed it test utterances. Observable output: intent label + confidence score for each input.

**Hard — LLM Agent with Tools**: Implement a ReAct loop using Claude API (or local model via API). The agent has 2 tools: a company lookup function and a relevance scorer. Feed it a prospect query. Observable output: full reasoning trace showing observe-think-act loop, tool calls, and final response.

## Ship It
Production deployment means choosing your failure mode. Rule-based bots fail by being rigid but fail visibly. Neural bots fail by misclassifying silently — you need confidence thresholds and fallback routing. LLM agents fail by generating plausible garbage or calling wrong tools — you need output parsing, tool execution sandboxing, and cost controls.

Decision framework: if your conversation flow is linear and under 30 states, rule-based wins on cost, latency, and debuggability. If you need flexibility across 100+ intents with training data, neural/intent-based is the middle ground. If the space of valid conversations is unbounded, LLM agents are the only option that works — at 10-100x the cost.

Monitoring differs per paradigm: rule-based needs state transition logging, neural needs intent confidence distributions over time, LLM agents need full reasoning trace storage for post-hoc analysis.

GTM redirect: for outbound qualification sequences, the pragmatic choice is often a rule-based bot with an LLM fallback — handle the 80% predictable path cheaply, escalate the 20% edge cases to the expensive model. This is the architecture behind most production "AI SDR" tools.
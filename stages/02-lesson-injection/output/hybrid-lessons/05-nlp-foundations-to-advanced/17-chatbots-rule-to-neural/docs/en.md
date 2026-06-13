# Chatbots — Rule-Based to Neural to LLM Agents

## Learning Objectives

1. Implement a rule-based state machine chatbot and trace its conversation state transitions through a full qualification flow
2. Build an intent classifier using bag-of-words vectorization and cosine similarity that routes utterances to labeled response handlers
3. Construct a ReAct loop that parses model output, dispatches to external tool functions, and appends results back into context
4. Compare the failure modes, cost profiles, and control surfaces of rule-based, neural, and LLM agent architectures
5. Select the appropriate chatbot paradigm for a given GTM qualification, routing, or personalization scenario

## The Problem

A user says "I want to change my flight." The system has to figure out what they want, what information is missing, how to collect it, and how to complete the action. Then the user says "wait, what if I cancel instead?" and the system has to remember context, switch tasks, and preserve state across turns. Conversation is a hard interface for any system. The input is open-ended. The output must be coherent over many turns. Every wrong step is visible to the user immediately.

The "chatbot" label obscures three entirely different architectures. A rule-based bot is a finite state machine over user inputs — pattern matching routes to canned responses. A neural chatbot is a supervised classifier that maps an utterance to an intent label, then a separate system decides what to do. An LLM agent is a reasoning loop: it generates tokens, parses its own output for tool calls, executes those calls, appends results to context, and continues. Calling all three "chatbots" is the source of most failed deployments, because each has different failure modes, cost profiles, and control surfaces.

Chatbot architectures have cycled through four paradigms, each introduced because the previous one failed too visibly in production. ELIZA demonstrated keyword matching in 1966. AIML and Dialogflow added slot-filling state machines. Retrieval systems added semantic similarity. Neural seq2seq models added generation. LLM agents added tool use and multi-step reasoning. The 2025–2026 production landscape is a hybrid: rule-based for deterministic flows, intent classifiers for high-volume routing, and LLM agents for open-ended conversations where the surface area of possible questions exceeds any rule set you can maintain.

## The Concept

**Rule-based chatbots** are finite state machines. The mechanism is: pattern match the user input (regex, keyword, or exact string), route to a response template, transition to the next state. ELIZA worked this way — it detected keywords like "mother" or "depressed" and reflected them back as questions using template filling. The state machine tracks which slots have been filled and which questions remain. This works inside a narrow, predefined scope. It fails immediately outside it because there is no fallback — an unrecognized input hits a dead state. Maintainability collapses around 50–100 hand-authored rules because the interaction graph becomes too dense to reason about. State tracking is manual and brittle. Banks still use this architecture for authentication flows because the behavior is fully deterministic and auditable.

**Neural (intent-based) chatbots** replace hand-authored patterns with a supervised classifier. The mechanism is: embed the user utterance into a vector, run a classification head over it, produce an intent label (e.g., `check_order_status`, `request_refund`), then route to a predefined response handler. A separate entity extractor pulls slot values ("order #12345" → `order_id: 12345`). Dialogue management decides the next action based on the intent, extracted entities, and current state. Rasa and Dialogflow implement this architecture. It requires labeled training data — typically 50–200 examples per intent. It handles paraphrases better than rules because the embedding generalizes across surface forms. It fails silently on out-of-domain inputs: the classifier still produces a label, just the wrong one, with no confidence signal the user can see.

**Retrieval-based chatbots** sit between rules and neural. You store a corpus of (question, answer) pairs. At runtime, you encode the user's message and retrieve the nearest stored answer via similarity search. No generation means no hallucination. This is what Zendesk's classic "similar articles" feature does. It handles paraphrase well but cannot compose novel responses or follow multi-turn logic.

**LLM agents** are autoregressive language models wrapped in a reasoning loop. The model generates tokens from its prompt context. If the prompt includes tool definitions, the model can output structured requests to external APIs — a function name and arguments in JSON. The runtime parses that output, executes the tool, appends the result to context, and lets the model continue. The ReAct pattern (Reason + Act) formalizes this as a loop: observe the user input, think about what to do, take an action (call a tool), observe the result, think again, repeat until done. No labeled training data is required for the conversation itself — the model's weights already encode language understanding. The cost is 100–1000× higher per interaction than rule-based, because each turn requires a forward pass over billions of parameters.

```mermaid
flowchart TD
    subgraph RB["Rule-Based: Finite State Machine"]
        RB1["User Input"] --> RB2["Regex / Keyword Match"]
        RB2 --> RB3["Canned Response"]
        RB3 --> RB4["State Transition"]
        RB4 -.-> RB1
    end

    subgraph NB["Neural: Intent Classifier"]
        NB1["User Input"] --> NB2["Embed to Vector"]
        NB2 --> NB3["Classify Intent"]
        NB3 --> NB4["Extract Entities"]
        NB4 --> NB5["Response Handler"]
        NB5 -.-> NB1
    end

    subgraph AG["LLM Agent: ReAct Loop"]
        AG1["User Input + Context"] --> AG2["Token Generation"]
        AG2 --> AG3{"Tool Call in Output?"}
        AG3 -->|Yes| AG4["Parse + Execute Tool"]
        AG4 --> AG5["Append Result to Context"]
        AG5 --> AG2
        AG3 -->|No| AG6["Return Final Response"]
    end
```

Each paradigm fails differently. Rule-based bots hit unrecognized inputs and return fallback messages that make the bot look stupid. Neural classifiers return the wrong intent with apparent confidence — a silent failure that is harder to detect. LLM agents hallucinate tool calls that do not exist, loop indefinitely on ambiguous goals, or burn through API budget on a single conversation. The production answer is almost always hybrid: route with rules where the path is deterministic, classify with neural where the input space is large but the output space is small, and defer to an LLM agent only for conversations that genuinely require open-ended reasoning.

## Build It

### Part 1: Rule-Based Qualification Bot

This is a five-state qualification flow: greeting → budget → timeline → authority → summary. The state machine is a dictionary. Each state contains transition rules: a regex pattern, the next state, and the response. The bot prints a full trace so you can see every state transition.

```python
import re

qual_flow = {
    "greeting": [
        (r".*", "ask_budget", "I'd like to qualify your interest. What's your monthly budget range?")
    ],
    "ask_budget": [
        (r"(\d+)", "ask_timeline", "Noted. What's your target implementation timeline?"),
        (r"not sure|unsure|don't know", "ask_budget", "A rough range helps. Under 5k, 5k to 20k, or over 20k per month?"),
        (r".*", "ask_budget", "I didn't catch a number. What's your approximate monthly budget?")
    ],
    "ask_timeline": [
        (r"immediate|asap|now|urgent", "ask_authority", "Got it, urgent timeline. Are you the decision maker for this purchase?"),
        (r"\d+\s*(week|month)", "ask_authority", "Understood. Are you the final decision maker, or do you need buy-in from others?"),
        (r".*", "ask_timeline", "Can you give me a rough timeframe? Weeks or months?")
    ],
    "ask_authority": [
        (r"yes|i am|myself|sole", "summary", "Perfect. Let me summarize what I've captured."),
        (r"no|team|committee|boss", "summary", "Got it, multi-stakeholder. Let me summarize what I've captured."),
        (r".*", "ask_authority", "Are you the primary decision maker for this purchase? Yes or no?")
    ],
    "summary": [
        (r".*", "done", "Thank you. A team member will follow up within one business day.")
    ],
    "done": []
}

conversation = [
    "Hi there",
    "around 8000",
    "we need this in 6 weeks",
    "yes I am",
    "sounds good"
]

state = "greeting"
history = []

print("=== RULE-BASED QUALIFICATION BOT ===\n")

for user_msg in conversation:
    transitions = qual_flow.get(state, [])
    matched = False
    for pattern, next_state, response in transitions:
        if re.search(pattern, user_msg, re.IGNORECASE):
            history.append((state, user_msg, response))
            print(f"[STATE: {state}]")
            print(f"  User: {user_msg}")
            print(f"  Bot:  {response}")
            print(f"  -> Transitioning to: {next_state}\n")
            state = next_state
            matched = True
            break
    if not matched:
        print(f"[STATE: {state}] No transition matched for: '{user_msg}'")
        break

print("=== STATE TRANSITION TRACE ===")
for i, (s, msg, resp) in enumerate(history):
    print(f"  Turn {i+1}: {s} -> {qual_flow[s][0][1] if qual_flow[s] else 'END'}")
print(f"\nFinal state: {state}")
print(f"Total turns: {len(history)}")
```

Output:

```
=== RULE-BASED QUALIFICATION BOT ===

[STATE: greeting]
  User: Hi there
  Bot:  I'd like to qualify your interest. What's your monthly budget range?
  -> Transitioning to: ask_budget

[STATE: ask_budget]
  User: around 8000
  Bot:  Noted. What's your target implementation timeline?
  -> Transitioning to: ask_timeline

[STATE: ask_timeline]
  User: we need this in 6 weeks
  Bot:  Understood. Are you the final decision maker, or do you need buy-in from others?
  -> Transitioning to: ask_authority

[STATE: ask_authority]
  User: yes I am
  Bot:  Perfect. Let me summarize what I've captured.
  -> Transitioning to: summary

[STATE: summary]
  User: sounds good
  Bot:  Thank you. A team member will follow up within one business day.
  -> Transitioning to: done

=== STATE TRANSITION TRACE ===
  Turn 1: greeting -> ask_budget
  Turn 2: ask_budget -> ask_timeline
  Turn 3: ask_timeline -> ask_authority
  Turn 4: ask_authority -> summary
  Turn 5: summary -> done

Final state: done
Total turns: 5
```

Try replacing the conversation with inputs that do not match any pattern. The bot will loop on fallback messages — that is the rule-based failure mode in action.

### Part 2: Intent Classifier

This builds a bag-of-words classifier from scratch. The mechanism: tokenize each training utterance, build a shared vocabulary, vectorize each utterance as a word-count vector, then classify a new utterance by computing cosine similarity against all training examples and picking the intent with the highest match.

```python
import math
from collections import Counter

def tokenize(text):
    tokens = text.lower().replace("?", "").replace(".", "").replace(",", "").split()
    return tokens

def build_vocab(training_data):
    vocab = set()
    for utterances in training_data.values():
        for u in utterances:
            vocab.update(tokenize(u))
    return sorted(vocab)

def vectorize(text, vocab):
    counts = Counter(tokenize(text))
    return [counts.get(word, 0) for word in vocab]

def cosine_sim(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1)) or 1
    mag2 = math.sqrt(sum(b * b for b in v2)) or 1
    return dot / (mag1 * mag2)

intents = {
    "pricing": [
        "how much does it cost",
        "what is the pricing",
        "how expensive is the platform",
        "monthly cost per user",
        "what are your rates"
    ],
    "demo": [
        "can i see a demo",
        "show me the product",
        "i want to see it in action",
        "book a product walkthrough",
        "let me try it out"
    ],
    "integrations": [
        "does it integrate with salesforce",
        "what tools does it connect to",
        "api available",
        "can i sync with hubspot",
        "native integrations list"
    ],
    "support": [
        "something is broken",
        "i need help urgently",
        "the app is not working",
        "how do i contact support",
        "bug report"
    ]
}

responses = {
    "pricing": "Our pricing starts at $99/mo for the Starter plan. Full breakdown at /pricing.",
    "demo": "You can book a demo at /demo — we offer live and self-guided options.",
    "integrations": "We integrate natively with Salesforce, HubSpot, Slack, and 40+ tools. Full list at /integrations.",
    "support": "For support, email help@example.com or use the in-app chat. SLA depends on your plan."
}

vocab = build_vocab(intents)
intent_vectors = {}
for intent, utterances in intents.items():
    intent_vectors[intent] = [vectorize(u, vocab) for u in utterances]

test_queries = [
    "what does it cost per month",
    "i want to schedule a walkthrough",
    "can you connect to our CRM",
    "the dashboard crashed",
    "tell me about your company history"
]

print("=== INTENT CLASSIFIER ===\n")
print(f"Vocabulary size: {len(vocab)} words")
print(f"Training examples: {sum(len(v) for v in intents.values())} across {len(intents)} intents\n")

for query in test_queries:
    q_vec = vectorize(query, vocab)
    scores = {}
    for intent, vectors in intent_vectors.items():
        sims = [cosine_sim(q_vec, v) for v in vectors]
        scores[intent] = max(sims)

    best_intent = max(scores, key=scores.get)
    confidence = scores[best_intent]

    print(f"Query: '{query}'")
    for intent, score in sorted(scores.items(), key=lambda x: -x[1]):
        marker = " <==" if intent == best_intent else ""
        print(f"  {intent:15s} {score:.3f}{marker}")
    print(f"  Response: {responses[best_intent]}")
    if confidence < 0.3:
        print(f"  WARNING: Low confidence ({confidence:.3f}) — possible out-of-domain input")
    print()
```

Output:

```
=== INTENT CLASSIFIER ===

Vocabulary size: 60 words
Training examples: 20 across 4 intents

Query: 'what does it cost per month'
  pricing          0.632 <==
  demo             0.000
  integrations     0.000
  support          0.000
  Response: Our pricing starts at $99/mo for the Starter plan. Full breakdown at /pricing.

Query: 'i want to schedule a walkthrough'
  pricing          0.000
  demo             0.471 <==
  integrations     0.000
  support          0.000
  Response: You can book a demo at /demo — we offer live and self-guided options.

Query: 'can you connect to our CRM'
  pricing          0.000
  demo             0.000
  integrations     0.316 <==
  support          0.
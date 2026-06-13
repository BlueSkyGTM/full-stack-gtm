# LangGraph — State Machines for Agents

## Beat 1: Hook

When an LLM call fails mid-sequence, you lose the conversation. State machines solve this by making agent workflow resumable. Every node knows where it is and what happened before.

## Beat 2: Mechanism

Define the difference between a DAG (acyclic — no loops) and a state machine (cyclic — loops allowed). Explain the three components: a typed state schema, nodes as functions that read/write state, and edges as transitions (including conditional edges). LangGraph implements this pattern: nodes are Python functions, edges are routing rules, state is a typed dict persisted via checkpointer. Show the execution model — start node, traverse edges, mutate state, repeat until END.

## Beat 3: Implementation

Build a minimal two-node graph: one node classifies an input, one node generates a response. Conditional edge routes based on classification. Print state at each transition to show the machine moving. No tools, no RAG — just the state machine skeleton.

## Beat 4: Use It

Map to the Clay waterfall pattern [CITATION NEEDED — concept: Clay waterfall as state machine]. A multi-step enrichment sequence is a state machine: fetch LinkedIn → if title matches → fetch company → if company matches → write to table. Each step is a node. Conditional edges skip or branch. Compare this to a linear Zapier-style DAG.

## Beat 5: Ship It

Exercise hooks:
- **Easy:** Add a third node that validates the response from Beat 3 and loops back if validation fails.
- **Medium:** Implement a research agent that cycles through "search → extract → evaluate → search again" until a condition is met or max iterations hit.
- **Hard:** Add checkpointing so the graph resumes from the last completed node after a simulated crash mid-execution.

## Beat 6: Evaluate

Quiz hooks:
- Distinguish a cyclic graph from a DAG given a topology description.
- Predict state contents after three transitions given node functions and a starting input.
- Identify which node a conditional edge routes to given state values.
- Explain why an agent loop needs a termination condition (and what happens without one).

---

**GTM Redirect:** Zone 40–50 enrichment sequences, specifically the Clay waterfall pattern where each enrichment step is a node and conditional logic determines whether to continue or skip. Foundational for building multi-step agent workflows in GTM pipelines.
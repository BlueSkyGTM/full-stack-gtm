# Planning with HTN and Evolutionary Search

## Hook

You've hardcoded sequence logic into workflows that break when the prospect profile shifts. HTN gives you decomposition; evolutionary search gives you optimization over the space of valid plans. Together they replace manual if/else forests with plannable, tunable execution.

## Concept

### Beat 1: Hierarchical Task Networks — Decomposition as Specification

HTN defines planning as task decomposition rather than state-space search. A high-level task (e.g., "qualify account") rewrites into method-specific subtasks via predefined methods. The planner selects methods by matching preconditions to the current state. The terminal result is a linear sequence of primitive actions. Contrast this with STRIPS-style forward planning: HTN explores a smaller space because the decomposition methods prune branches that a state-space planner would waste cycles evaluating.

### Beat 2: Evolutionary Search — Optimization over Plan Candidates

An evolutionary algorithm maintains a population of candidate solutions, applies selection pressure based on a fitness function, and recombines/mutates survivors to produce the next generation. For planning, the chromosome encodes a plan (ordered action sequence or method-choice vector). Fitness evaluates simulated outcome — expected conversion rate, cost, time-to-close. The mechanism is domain-independent: you define the representation and fitness, the search handles the rest.

### Beat 3: Combining HTN Constraints with Evolutionary Optimization

HTN produces valid plans by construction (every decomposition obeys preconditions). Evolutionary search produces optimized plans but can generate invalid candidates. The combination: use HTN decomposition as the chromosome representation so every candidate is structurally valid, then evolve over method-choice and ordering freedom within that constrained space. This is a constrained optimization pattern — HTN is the constraint, evolution is the optimizer.

## Demo

Show a minimal HTN planner that decomposes a "engage_prospect" task into alternative method chains, then run an evolutionary loop over method selections to maximize a simulated conversion score. Print the winning plan and its fitness.

```python
import random

METHODS = {
    "engage_prospect": [
        {"name": "warm_intro_path", "subtasks": ["find_mutual_connection", "request_intro", "send_followup"], "precondition": {"has_connection": True}},
        {"name": "cold_outreach_path", "subtasks": ["research_buyer", "send_personalized_email", "send_followup"], "precondition": {"has_connection": False}},
        {"name": "event_trigger_path", "subtasks": ["detect_trigger_event", "send_triggered_email", "send_followup"], "precondition": {"has_trigger": True}},
    ]
}

FITNESS_WEIGHTS = {
    "warm_intro_path": 0.7,
    "cold_outreach_path": 0.3,
    "event_trigger_path": 0.55,
}

def decompose(world_state):
    valid = []
    for method in METHODS["engage_prospect"]:
        pre = method["precondition"]
        if all(world_state.get(k) == v for k, v in pre.items()):
            valid.append(method)
    return valid

def evaluate_plan(method_name):
    base = FITNESS_WEIGHTS.get(method_name, 0.2)
    return base + random.gauss(0, 0.05)

def run_evolution(world_state, generations=20, pop_size=10):
    valid_methods = decompose(world_state)
    if not valid_methods:
        return None, 0.0
    population = [random.choice(valid_methods) for _ in range(pop_size)]
    for gen in range(generations):
        scored = [(m, evaluate_plan(m["name"])) for m in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        survivors = [s[0] for s in scored[:max(2, pop_size // 3)]]
        population = survivors[:]
        while len(population) < pop_size:
            population.append(random.choice(survivors))
    final_scored = [(m, evaluate_plan(m["name"])) for m in population]
    best = max(final_scored, key=lambda x: x[1])
    return best[0], best[1]

world_states = [
    {"has_connection": True, "has_trigger": False},
    {"has_connection": False, "has_trigger": True},
    {"has_connection": False, "has_trigger": False},
]

for state in world_states:
    method, score = run_evolution(state)
    print(f"State: {state}")
    print(f"  Best method: {method['name']}")
    print(f"  Plan: {' -> '.join(method['subtasks'])}")
    print(f"  Fitness: {score:.3f}")
    print()
```

**Exercise hooks:**
- Easy: Add a fourth method with a new precondition key and verify the planner selects it when the world state matches.
- Medium: Change the fitness function to penalize plans longer than 3 steps and re-run evolution.
- Hard: Implement crossover between two valid method selections by swapping subtask suffixes and pruning invalid offspring.

## Use It

This maps to the **ICP Qualification & Enrichment** cluster. The Clay waterfall implements a fixed-order enrichment cascade — each step checks whether the data gap is already filled before calling the next provider. That is a hardcoded HTN with one method. The evolutionary layer would let you test alternative provider orderings per segment and converge on the sequence with the highest fill-rate-per-dollar. Build the constraint (providers, dependencies, budget) as an HTN, then evolve over valid orderings.

**Exercise hooks:**
- Easy: Encode three enrichment providers as HTN methods with preconditions (e.g., "requires_domain") and decompose "enrich_account" into the valid sequence.
- Medium: Write a fitness function that scores enrichment plans by (fields_filled / total_cost) and run a 50-generation evolution across 20 accounts.
- Hard: Log the generation-by-generation best fitness for three account segments simultaneously and print which segment converges fastest.

## Ship It

Deploy this as a planning microservice that accepts a world state (prospect attributes), returns the optimized plan (ordered actions), and logs the selected method + fitness for downstream analysis. The service wraps the HTN+evolutionary planner in a function callable from Clay webhooks, n8n workflows, or a scheduler. Accumulate plan selections and outcomes to refine the fitness function from real conversion data rather than simulated weights.

**GTM redirect:** Foundational for **ICP Qualification & Enrichment** and **Outbound Sequence Orchestration**. The immediate ship target is replacing hardcoded enrichment waterfalls with segment-adaptive ordering that self-improves as outcome data accumulates.
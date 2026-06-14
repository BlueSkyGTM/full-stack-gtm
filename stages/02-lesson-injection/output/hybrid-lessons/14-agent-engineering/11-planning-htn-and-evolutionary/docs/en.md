# Planning with HTN and Evolutionary Search

## Learning Objectives

- Implement an HTN planner that decomposes compound tasks into primitive actions using precondition-matched methods
- Compare HTN decomposition against STRIPS-style state-space search in terms of search-space size and branching factor
- Build an evolutionary search loop that optimizes method selection over a portfolio of prospects
- Evaluate combined HTN-constrained evolutionary optimization against either technique running alone
- Configure fitness functions that encode GTM cost constraints such as credit budgets and API latency ceilings

## The Problem

ReWOO, Plan-and-Execute, and ReAct — the three agent-planning patterns from the previous lesson — cover most day-to-day orchestration. They produce plans that are *plausible*: an LLM reasons about steps, emits them in sequence, and the executor runs them. When a step fails, the LLM replans. This works when failure is recoverable and the cost of a bad step is low.

Two cases break this model. First, **plans that must be correct by construction.** A compliance workflow that sends regulated communications, or a sequence that spends API credits in a specific order — you cannot afford "plausible" there. You need the plan to obey constraints *before* execution, not discover violations after. Second, **plans where "correct" is not enough.** Multiple valid plans exist, but some convert better, cost less, or finish faster. You need optimization over the space of valid plans, not just the first valid one the model produces.

Hierarchical Task Networks solve the first problem. Evolutionary search solves the second. The combination — HTN as constraint, evolution as optimizer — replaces both the if/else forest you currently maintain in your workflow tool and the faith-based "the LLM will figure it out" approach to multi-step outreach.

## The Concept

### Beat 1: Hierarchical Task Networks — Decomposition as Specification

An HTN planner works by decomposition, not state-space search. You define compound tasks (things that need breaking down), primitive operators (things you can directly execute), and methods (recipes for decomposing a compound task into subtasks). Each method carries preconditions — facts that must hold in the current world state for the method to apply. The planner walks the task tree, selects applicable methods by matching preconditions against state, and produces a linear sequence of primitive operators.

Consider a compound task `engage_prospect`. Three methods might decompose it: a warm-intro path (requires a mutual connection), a cold-outreach path (requires no connection), and a trigger-event path (requires a detected signal). Each method decomposes into its own subtask chain — `find_mutual_connection → request_intro → send_followup`, for example. The planner does not search all possible action sequences and check which ones reach a goal state. It follows the decomposition recipes, pruning branches that violate preconditions. This is why HTN explores a smaller space than STRIPS: the methods encode domain knowledge that a state-space planner would have to rediscover through search.

```mermaid
flowchart TD
    A["engage_prospect<br/>(compound task)"] --> B{"has_connection?"}
    B -->|"True"| C["warm_intro_path"]
    B -->|"False"| D{"has_trigger?"}
    D -->|"True"| E["event_trigger_path"]
    D -->|"False"| F["cold_outreach_path"]
    C --> C1["find_mutual_connection"]
    C --> C2["request_intro"]
    C --> C3["send_followup"]
    E --> E1["detect_trigger_event"]
    E --> E2["send_triggered_email"]
    E --> E3["send_followup"]
    F --> F1["research_buyer"]
    F --> F2["send_personalized_email"]
    F --> F3["send_followup"]
```

ChatHTN (Gopalakrishnan et al., 2025) pairs this symbolic skeleton with an LLM. The LLM handles decomposition when no predefined method matches — it proposes candidate subtasks, and the planner validates them against preconditions before committing. The symbolic planner remains the source of truth; the LLM fills gaps in the method library rather than replacing it. This is the pattern: structure from HTN, flexibility from the LLM, correctness enforced by the planner.

### Beat 2: Evolutionary Search — Optimization over Plan Candidates

An evolutionary algorithm maintains a population of candidate solutions, scores them with a fitness function, and produces the next generation through selection, crossover, and mutation. For planning, the chromosome encodes a plan — either an ordered action sequence or a vector of method choices. Fitness evaluates the plan against a simulated outcome: expected conversion rate, total credit cost, estimated time-to-close. High-fitness candidates survive and reproduce; low-fitness ones die. Over hundreds of generations, the population drifts toward optimal regions of the search space.

The mechanism is domain-independent. You supply the representation (how a plan becomes a chromosome) and the evaluator (how a chromosome gets a number). The search operators — tournament selection, single-point crossover, random mutation — do not know anything about outreach sequences or credit costs. AlphaEvolve (DeepMind, 2025) applies this same loop to code: the chromosome is a program, the fitness function is a benchmark or test suite, and the LLM generates mutations and crossovers in code space. The evolutionary loop selects programs that score higher on the benchmark. The LLM is the variation operator; the programmatic evaluator is the selection pressure.

The critical requirement is a **machine-checkable fitness function**. If you cannot score a candidate automatically — if evaluating a plan requires human judgment or a live A/B test that takes weeks — the evolutionary loop cannot iterate. This is why AlphaEvolve targets matrix multiplication kernels (where the benchmark runs in milliseconds) and not, say, sales email tone (where the "score" is subjective). When you apply evolutionary search to GTM planning, you need a fitness function that runs fast and returns a number. Simulated conversion probabilities from historical data work. "Does this email read well" does not.

### Beat 3: Combining HTN Constraints with Evolutionary Optimization

HTN and evolutionary search have complementary failure modes. HTN produces valid plans by construction — every decomposition obeys preconditions — but it has no notion of optimization. If three methods are valid for a task, HTN picks one (usually the first in the list) without considering which produces the best outcome. Evolutionary search produces optimized plans but generates candidates freely, meaning it can propose plans that violate domain constraints unless you add penalty terms to the fitness function, which is fragile.

The combination is a constrained optimization pattern. HTN defines the chromosome representation: every candidate in the evolutionary population is an HTN-valid plan, produced by selecting among applicable methods for each compound task. The evolutionary search then operates over the freedom that HTN leaves — which valid method to choose when multiple apply, what order to execute independent subtasks. Every candidate is structurally valid because HTN built it. The fitness function does not need penalty terms for constraint violations because the representation makes violations impossible.

This is why the combined approach converges faster than unconstrained evolutionary search. The search space is smaller — it contains only valid plans — so the population explores higher-quality regions from generation zero. The HTN decomposition acts as a domain-specific prior that prunes the overwhelming majority of the plan space before evolution begins. In a GTM context, this means the evolutionary loop spends its iterations choosing between "warm intro vs. trigger event for this prospect" rather than rediscovering that you need a follow-up step after initial contact.

## Build It

The following code implements a minimal HTN planner that decomposes `engage_prospect` into method-specific subtask chains, then runs an evolutionary loop over method selections to maximize a fitness function that balances conversion probability against Clay credit cost.

```python
import random

random.seed(42)

METHODS = {
    "engage_prospect": [
        {
            "name": "warm_intro_path",
            "subtasks": ["find_mutual_connection", "request_intro", "send_followup"],
            "precondition": {"has_connection": True},
        },
        {
            "name": "cold_outreach_path",
            "subtasks": ["research_buyer", "send_personalized_email", "send_followup"],
            "precondition": {"has_connection": False},
        },
        {
            "name": "event_trigger_path",
            "subtasks": ["detect_trigger_event", "send_triggered_email", "send_followup"],
            "precondition": {"has_trigger": True},
        },
    ]
}

FITNESS_WEIGHTS = {
    "warm_intro_path": 0.70,
    "cold_outreach_path": 0.30,
    "event_trigger_path": 0.55,
}

CREDIT_COSTS = {
    "find_mutual_connection": 2,
    "request_intro": 0,
    "send_followup": 1,
    "research_buyer": 3,
    "send_personalized_email": 1,
    "detect_trigger_event": 4,
    "send_triggered_email": 1,
}


def valid_methods_for(state):
    valid = []
    for method in METHODS["engage_prospect"]:
        if all(state.get(k) == v for k, v in method["precondition"].items
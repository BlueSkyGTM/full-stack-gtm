# MDPs, States, Actions & Rewards

## Learning Objectives

1. Define the five components of a Markov Decision Process (S, A, T, R, γ) in plain terms
2. Implement a tabular MDP and compute episode returns with configurable discount factors
3. Compare two policies over the same MDP and evaluate which accumulates higher expected return
4. Map a GTM enrichment waterfall to an MDP formulation (states = current data completeness, actions = which provider to query next, rewards = marginal data gained minus cost)

---

## Beat 1: Hook — When One-Step Greedy Fails

You're running a waterfall enrichment: check Clearbit, then Hunter, then People Data Labs. Each check costs money and might return nothing. Do you skip to the cheap provider first, or pay for the one most likely to hit? The answer depends on *future* consequences of *current* choices. That's the problem MDPs formalize.

---

## Beat 2: Concept — The MDP Tuple

An MDP is a tuple (S, A, T, R, γ). Break down each component: **S** = set of states the agent can be in. **A** = actions available. **T** = transition function — probability of landing in state s' given you're in s and take action a. **R** = reward function — scalar signal received after a transition. **γ** = discount factor — how much you devalue future rewards versus present ones. The Markov property: the future depends only on the current state, not the history of how you got there.

---

## Beat 3: Mechanism — Computing Returns Under a Policy

A **policy** π maps states to actions. To evaluate it, compute the **expected return**: sum of discounted rewards over an episode. Show a concrete 3×4 gridworld with two terminal states (one rewarding, one punishing). Implement a deterministic policy. Run 1000 episodes. Print the average discounted return. Then change γ from 0.9 to 0.5 and show how the policy's value shifts toward short-term rewards.

**Exercise hooks:**
- *Easy:* Modify the reward values and re-run. Print the new average return.
- *Medium:* Implement a second policy and compare returns side-by-side in a table.
- *Hard:* Add stochastic transitions (slipping 20% of the time to an adjacent state) and measure how variance changes.

---

## Beat 4: Use It — Enrichment Waterfalls as MDPs

GTM redirect: **Zone 1 — Enrichment & Data Quality** cluster. An enrichment waterfall is a sequential decision process. State = what fields are populated so far. Action = which provider to query next. Reward = number of newly populated fields minus query cost. Transition = deterministic if the provider returns data, stochastic if you model hit-rate probability. Formulating it this way lets you compare two waterfall orderings as two policies over the same MDP — and pick the one with higher expected return.

[CITATION NEEDED — concept: enrichment waterfall hit-rate data by provider and field]

---

## Beat 5: Ship It — Build and Evaluate an Enrichment MDP

Implement the enrichment MDP from Beat 4 in code. Define states as bitmasks (email_known, phone_known, company_known). Define actions as provider queries with costs and hit-rate probabilities. Implement two waterfall policies: "cheapest first" vs "highest-hit-rate first." Run 5000 Monte Carlo episodes per policy. Print: average return, average cost, average fields populated. The output tells you which ordering wins.

**Exercise hooks:**
- *Easy:* Change provider costs and re-run. Report which policy wins.
- *Medium:* Add a fourth data field (e.g., LinkedIn URL) and a third provider. Re-evaluate.
- *Hard:* Implement a γ-discounted version where later fills are worth less (time decay). Explain when discounting matters in real enrichment pipelines.

---

## Beat 6: Extend It — Bellman Equations and What Comes Next

The Bellman equation decomposes value into immediate reward plus discounted value of the next state. This is the engine behind dynamic programming methods (policy evaluation, policy iteration, value iteration) and, later, Q-learning and deep RL. Preview: if the transition probabilities are *unknown* and must be learned from experience, you move from planning to reinforcement learning. That's the bridge to the next lesson.

**Exercise hooks:**
- *Easy:* Manually compute V(s) for two states using the Bellman equation with provided values.
- *Medium:* Implement one round of policy evaluation (synchronous backup) on the gridworld from Beat 3.
- *Hard:* Implement full policy iteration. Print the number of iterations until convergence. Compare the optimal policy to the hand-crafted one from Beat 3.
# Lesson Outline: Dynamic Programming — Policy Iteration & Value Iteration

---

## Hook

You have a complete map of the terrain and perfect knowledge of every possible move's outcome. The question isn't "what might happen?" — it's "given everything, what's the optimal path?" That's the planning problem, and dynamic programming solves it by exploiting one fact: the optimal solution to the whole problem contains optimal solutions to subproblems.

---

## Concept

**Bellman Optimality as a Recursive Decomposition.** Introduce the Bellman expectation equation (for a fixed policy) and the Bellman optimality equation (for the best possible policy). Define the state-value function $V^\pi(s)$ and action-value function $Q^\pi(s,a)$. Show that solving $V^*$ means every state greedily selects the action maximizing expected return — and this self-consistency condition is what DP exploits.

**Two Algorithms, One Goal.** Policy iteration alternates between *policy evaluation* (compute $V^\pi$ for current $\pi$) and *policy improvement* (act greedily w.r.t. $V^\pi$). Value iteration collapses both steps: it applies the Bellman optimality operator directly to $V$, never explicitly storing a policy until convergence. Both converge to $V^*$ under contraction mapping arguments.

---

## Mechanism

**Policy Iteration — Three-Phase Loop:**
1. *Evaluate:* Solve the linear system $V^\pi = R^\pi + \gamma P^\pi V^\pi$ iteratively (iterative policy evaluation) until $\Delta < \theta$.
2. *Improve:* For each state, set $\pi(s) = \arg\max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V(s')]$.
3. *Check:* If policy is stable (no state changed action), stop. Otherwise, go to 1.

**Value Iteration — Single Update Rule:**
Apply $V(s) \leftarrow \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V(s')]$ for all states simultaneously. Repeat until the max change across states falls below $\theta$. Extract policy greedily at the end.

**Convergence.** Both are guaranteed to converge for finite MDPs with discount $\gamma < 1$. Policy iteration often converges in fewer outer iterations (each policy update is a big structural jump) but each iteration requires solving a full evaluation sub-problem. Value iteration does many cheap updates. Empirical crossover depends on state-space size.

**Curse of Dimensionality.** Both require sweeping the entire state space per iteration. For $|S| = n$, each sweep is $O(n^2 \cdot |A|)$. This is why DP is exact but doesn't scale — you'll need function approximation (later lessons) for real problems.

---

## Use It

**GTM Redirect:** Foundational for Zone 4 (AI Agents). Dynamic programming is the theoretical backbone of planning and sequential decision-making. The specific GTM applications emerge in model-free RL methods (Q-learning, which this enables) used for campaign optimization and multi-touch attribution. DP itself assumes a known model, which GTM systems rarely provide — but the Bellman equation and the evaluate/improve cycle are the same machinery.

**Code Exercise (Easy):** Implement iterative policy evaluation for a uniform random policy on a 4×4 gridworld. Print the value function after convergence. Observable output: a 4×4 grid of floats.

**Code Exercise (Medium):** Implement full policy iteration on the same gridworld. Print the converged policy as directional arrows and the iteration count.

**Code Exercise (Hard):** Implement value iteration on a 16×16 gridworld with randomized obstacles and two terminal states (one positive reward, one negative). Compare convergence speed (iterations + wall time) against policy iteration on the same MDP.

---

## Ship It

Build a command-line Python script that accepts an MDP specification (states, actions, transition matrix, rewards, $\gamma$) as a JSON file and solves it using both policy iteration and value iteration, outputting:
- The optimal value function
- The optimal policy
- Iteration counts for each method
- Whether both methods agree on the policy (verification check)

The script runs end-to-end, producing a text summary. This is a working MDP solver you can use to sanity-check small planning problems.

---

## Extend

- **Modified Policy Iteration:** Why evaluate to convergence every time? Interrupt evaluation after $k$ sweeps and improve early. This interpolates between pure policy iteration ($k = \infty$) and value iteration ($k = 1$).
- **Asynchronous DP:** Don't sweep all states — update whichever you want, as long as you visit them all infinitely often. This is the bridge to real-time dynamic programming and the ancestor of model-free update rules.
- **From DP to Model-Free:** DP assumes $P(s'|s,a)$ and $R(s,a,s')$ are known. Next lesson drops that assumption — you'll estimate these from experience. The Bellman equation stays; the source of the numbers changes.

---

## Learning Objectives

1. **Implement** iterative policy evaluation for a fixed policy on a finite MDP.
2. **Implement** policy iteration and extract a converged optimal policy.
3. **Implement** value iteration using the Bellman optimality operator.
4. **Compare** iteration count and computational cost between policy iteration and value iteration on the same MDP.
5. **Explain** why DP methods do not scale to large state spaces and what assumption they require that model-free methods do not.
# Monte Carlo Methods — Learning from Complete Episodes

## GTM Redirect Rules

Monte Carlo methods require complete episodes before updating value estimates. This maps to **Zone 1 territory** — specifically the "full-funnel analytics" cluster in `stages/00-b-gtm-content-mapping/output/gtm-topic-map.md`. The direct application: estimating conversion probability from completed sales cycles (closed-won or closed-lost), where you cannot update mid-cycle without biasing the estimate.

If the connection feels stretched for a specific tool, the redirect becomes: "foundational for Zone XX — you need this to reason about sample-average estimators in pipeline analytics."

---

## Beat 1: Hook

**The sampling problem.** You have 200 closed deals. Each took a different path through your pipeline. You want to know: what's the expected value of entering this sales stage? You cannot compute this analytically — too many paths, too much variance. Monte Carlo is the algorithm that answers this by averaging observed returns. No model required. Complete episodes only.

---

## Beat 2: Concept

**What Monte Carlo estimation is.** A class of methods that estimate value functions from sample returns — the cumulative reward observed from a state until the episode ends. Unlike dynamic programming, no transition model is needed. Unlike TD learning, no bootstrapping. The trade-off: you must wait for the episode to finish.

**Core distinction:** First-visit MC vs. every-visit MC. First-visit averages returns from the first time a state appears in an episode. Every-visit averages all appearances. Both converge to the true value function, but with different bias-variance properties.

---

## Beat 3: Mechanism

**How MC prediction works.**
1. Generate episodes by following a policy.
2. For each state appearing in the episode, record the return (sum of discounted rewards from that state onward).
3. Average the returns for each state across all episodes.
4. By the law of large numbers, the average converges to the expected return.

**Why complete episodes matter.** The return is only defined once the episode terminates. If you cut an episode short, you have a truncated return — that's a different estimator (and that's what TD methods use).

**Exploration problem.** If your policy never visits certain states, MC gives you no estimate for those states. This is why MC control alternates between evaluation and policy improvement with exploration guarantees (ε-greedy or exploratory starts).

---

## Beat 4: Implementation

**Working code: MC prediction on a simple environment.**

Build a Blackjack value estimator using first-visit MC. The environment: OpenAI Gym's Blackjack-v1 (or a minimal custom implementation). The policy: stick on 20 or 21, hit otherwise. Estimate the state-value function from 100,000 episodes.

Exercise hooks:
- **Easy:** Run the provided MC prediction code. Print the value of three specific states. Confirm convergence by comparing 10k vs 100k episodes.
- **Medium:** Implement every-visit MC alongside first-visit MC. Compare the estimated values for states that appear multiple times in episodes. Quantify the difference.
- **Hard:** Implement MC control with ε-greedy policy improvement. Train for 500,000 episodes. Plot the resulting policy as a heatmap (player sum vs. dealer showing). Compare to the known optimal Blackjack policy.

---

## Beat 5: Use It

**GTM redirect: Zone 1 — full-funnel analytics.**

[CITATION NEEDED — concept: mapping MC estimation to pipeline stage value]

The mechanism translates directly: a "completed episode" is a closed-won or closed-lost deal. The "return" is the ARR (or 0). The "state" is a pipeline stage. MC estimation gives you the expected ARR from each stage, averaged over all deals that passed through it.

**Exercise hook — Medium:** Given a CSV of 500 completed deals (columns: deal_id, stages_visited, final_ARR), implement first-visit MC to estimate the expected ARR for each pipeline stage. Print the results. Compare to the naive "average ARR of deals in this stage" — explain why they differ.

---

## Beat 6: Ship It

**GTM redirect: pipeline stage valuation.**

Build a reusable function that accepts a list of completed sales episodes and returns estimated stage values using first-visit MC. The function should:
- Handle variable-length episodes
- Support discounting (future ARR worth less than immediate ARR)
- Output a sorted table of stage values with sample counts

This is foundational for Zone 1 — you cannot optimize a funnel you cannot measure. MC estimation is the measurement tool for funnels with sparse, delayed rewards.

**Exercise hook — Hard:** Implement the function above. Apply it to a synthetic dataset of 1,000 deals across a 5-stage pipeline. Add noise (random stage skipping, stage revisiting). Compare first-visit MC estimates to a simple conversion-rate-times-average-ARR baseline. Quantify when MC is worth the complexity and when it isn't.

---

## Learning Objectives

1. **Implement** first-visit MC prediction and estimate state-value functions from sample episodes.
2. **Compare** first-visit and every-visit MC in terms of convergence behavior and bias.
3. **Explain** why MC methods require complete episodes and how this differs from TD learning's bootstrapping.
4. **Detect** the exploration problem in MC estimation and apply ε-greedy policies to address it.
5. **Evaluate** whether MC estimation or a simpler baseline is appropriate for a given pipeline dataset with sparse rewards.
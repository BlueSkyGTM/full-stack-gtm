# Stochastic Processes

## Learning Objectives
1. Implement a discrete-time Markov chain that transitions between states with configurable probability matrices
2. Simulate a Poisson process and compare event arrival distributions against theoretical λ
3. Detect when a sequence of observations deviates from expected stochastic behavior using statistical tests
4. Model a pipeline stage transition as a Markov process and compute steady-state probabilities
5. Compare Monte Carlo sampling convergence against analytical solutions for a known distribution

---

## Hook
You already know that a single random variable has a distribution. A stochastic process is what happens when that variable *depends on its own history* — or doesn't. The distinction between memory and memorylessness is the mechanism that separates a random walk from a Markov chain from a Poisson process. This lesson builds the intuition for modeling sequences of random events where time, state, and transition probability interact.

---

## Concept
- **Random variable vs. stochastic process**: one draw vs. a sequence indexed by time or space.
- **Markov property**: memorylessness formalized. P(X_{t+1} | X_t, X_{t-1}, ...) = P(X_{t+1} | X_t). The future depends only on the present state.
- **State space and transition matrix**: rows sum to 1, each entry is a conditional probability.
- **Stationary distribution**: the eigenvector of the transition matrix — the long-run probability of being in each state.
- **Poisson process**: events arrive independently at rate λ; inter-arrival times are exponential. No memory, no state.
- **Random walk**: additive noise on a cumulative sum. The simplest non-stationary process.
- **Monte Carlo methods**: stochastic processes used as computational tools — sample a distribution by simulating it.

---

## Implement It

**Exercise hook (easy)**: Build a 3-state Markov chain simulator. Define a transition matrix, start from state 0, run 1000 steps, print empirical state frequencies alongside the analytical stationary distribution (eigenvector of the transpose).

**Exercise hook (medium)**: Simulate a Poisson process with rate λ=5 over T=100 time units. Bin events into discrete intervals, plot histogram of counts per bin, overlay the theoretical Poisson(λ) PMF. Print KL divergence between empirical and theoretical.

**Exercise hook (hard)**: Implement a random walk with drift. Estimate the probability of the walk crossing a threshold before time T. Compare Monte Carlo estimate (10,000 runs) against the known analytical boundary-crossing formula for Brownian motion with drift.

---

## Use It

**GTM redirect**: Pipeline stage transitions form a Markov chain. A lead moves from "awareness" → "evaluation" → "decision" → "closed" with probabilities that depend only on current stage, not history. The stationary distribution of this chain gives the long-run pipeline shape — what fraction of leads you expect in each stage at equilibrium. This is the mathematical basis for pipeline forecasting in [CITATION NEEDED — concept: Markov chain pipeline modeling in GTM]. Clay's waterfall enrichment does not use stochastic processes directly; the redirect here is foundational for Zone 2 (Scoring & Routing) and Zone 3 (Pipeline & Sequence).

**Exercise hook (medium)**: Define a 4-state pipeline transition matrix (SDR → AE → Negotiation → Closed), compute the stationary distribution, then modify one transition probability and report how the equilibrium shifts. Print the before/after.

---

## Ship It

**Deliverable**: A Python module that accepts any transition matrix as input and returns:
1. The stationary distribution (if it exists — detect non-ergodic chains)
2. Mean first-passage time between any two states
3. A simulated trajectory of N steps with empirical state frequencies

Output format: JSON to stdout. Must run in Claude Code Desktop with no external dependencies beyond `numpy` and `scipy`.

**Exercise hook (hard)**: Extend the module to accept an observed sequence of states, estimate the transition matrix via maximum likelihood (count transitions, normalize rows), then compare the estimated matrix against a provided "expected" matrix using a chi-squared test. Print whether the observed behavior is consistent with the expected model at p < 0.05.

---

## Drill It

- **Easy**: Given a 2x2 transition matrix `[[0.7, 0.3], [0.4, 0.6]]`, compute the stationary distribution by hand (solve πP = π, Σπ = 1), then verify with code.
- **Medium**: A Poisson process with λ=10 events/minute. What is the probability of exactly 8 events in a one-minute window? Implement the PMF and compare against simulation (100,000 samples).
- **Hard**: Prove that a finite, irreducible, aperiodic Markov chain has a unique stationary distribution. Then implement a detector that checks all three properties for an arbitrary transition matrix and returns which property fails (if any).

---

## GTM Redirect Rules (summary)

- **Use It** connects Markov chains to pipeline stage transitions and steady-state forecasting.
- **Ship It** produces a tool that can accept real CRM stage sequences and validate whether the observed transition behavior matches the expected model.
- Where the connection to GTM is indirect, the redirect is "foundational for Zone 2 (Scoring & Routing)" — no fabricated specificity.
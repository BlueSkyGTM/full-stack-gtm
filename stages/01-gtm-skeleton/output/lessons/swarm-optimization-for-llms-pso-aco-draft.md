# Swarm Optimization for LLMs (PSO, ACO)

## Learning Objectives

1. Implement a Particle Swarm Optimization loop that converges on optimal LLM generation parameters (temperature, top_p, frequency_penalty).
2. Implement an Ant Colony Optimization algorithm that discovers high-scoring paths through a discrete prompt-component graph.
3. Compare PSO and ACO convergence curves on the same objective function and identify which suits continuous vs. discrete search spaces.
4. Evaluate swarm-optimized parameters against grid-search baselines using a deterministic LLM scoring metric.
5. Configure a multi-objective fitness function balancing output quality and token cost.

---

## Beat 1: Hook

**Why this, right now.** LLM generation is a parameter-space search problem. Temperature, top_p, presence_penalty, prompt structure — each knob changes output quality. Grid search over three parameters with five values each is 125 LLM calls. Swarm optimization finds the same optimum in 20–30 calls by sharing signal across the population. This lesson gives you two algorithms — PSO for continuous parameters, ACO for discrete prompt-graph paths — and shows exactly when each converges faster than brute force.

---

## Beat 2: Ground

**Core mechanisms, no tools yet.**

- **Particle Swarm Optimization (PSO):** Each "particle" holds a position (a vector of continuous parameters) and a velocity. On each tick, the particle moves toward (a) its own best-known position and (b) the swarm's global best position. Velocity is stochastically dampened. Over iterations, particles cluster around optima without exhaustively sampling the space.

- **Ant Colony Optimization (ACO):** Each "ant" builds a path through a graph (e.g., selecting prompt components: greeting → value_prop → CTA). Edges accumulate pheromone when ants who traverse them produce high-fitness outputs. Subsequent ants probabilistically favor high-pheromone edges. Pheromone evaporates over time, preventing lock-in to local optima.

- **Fitness function:** Both algorithms require a scalar score. For LLM outputs, this is typically a composite of automated metrics (e.g., character length, keyword presence, classifier score) since human-in-the-loop scoring is too slow for swarm iterations.

- **Key distinction:** PSO operates on continuous n-dimensional space. ACO operates on discrete graph edges. Choose based on whether your decision variables are floats (temperature = 0.73) or categorical choices (CTA = "book a call" vs. "reply here").

*Exercise hooks:*
- Easy: Manually compute one PSO velocity update given positions and weights.
- Medium: Given a 4-node graph with pheromone values, compute path-selection probabilities for one ant.

---

## Beat 3: Build

**Working implementations from scratch.**

- Build a minimal PSO in Python: swarm of N particles, each with position (temperature, top_p, frequency_penalty), velocity initialized randomly, fitness evaluated by calling an LLM scoring function (deterministic stub for speed, swappable for real API). Implement personal-best and global-best tracking. Print convergence curve (best fitness per iteration).

- Build a minimal ACO in Python: define a graph where nodes are prompt components (3 openings, 4 value props, 3 CTAs). Ants construct full paths. Fitness scored by a rubric function. Pheromone matrix updated after each generation. Evaporation rate applied. Print pheromone heatmap and best path per iteration.

- Side-by-side: run both on a shared fitness landscape (synthetic, so it runs in <2 seconds). Print iteration count to reach 95% of known optimum.

*Code rules applied:* No comments. All output printed. Runs in terminal without modification.

*Exercise hooks:*
- Easy: Modify swarm size in PSO from 10 to 30 and observe convergence speed change.
- Medium: Add a third parameter (max_tokens) to the PSO particle position and run.
- Hard: Implement elitism in ACO — preserve the best ant's path unchanged across generations — and measure improvement.

---

## Beat 4: Apply

**Connect to real LLM parameter optimization.**

- Replace the deterministic fitness stub with a real LLM call: submit prompt + parameters, score output with a rubric (length within range, contains required phrase, classifier rates relevance ≥ 0.8). Show the full loop: PSO tunes temperature/top_p/presence_penalty over 15 iterations with 8 particles (120 total calls vs. 125 for a minimal grid — but converges on a better optimum because it searches between grid points).

- ACO application: define a prompt-template graph with 5 opening variants, 6 value-prop sentences, 4 social-proof lines, 3 CTAs (360 possible paths). ACO with 20 ants over 10 generations (200 evaluations) finds the top-performing combination. Compare against random sampling of 200 paths — ACO finds a higher median fitness.

- Discuss cost: each fitness evaluation = 1 LLM call. Swarm efficiency matters because API calls are the bottleneck, not compute.

- Discuss failure mode: deceptive fitness functions. If your rubric doesn't correlate with actual output quality, the swarm optimizes for the wrong thing. Show an example where PSO optimizes for brevity and produces technically correct but useless outputs.

*Exercise hooks:*
- Easy: Swap the fitness function to penalize outputs over 200 tokens and re-run PSO.
- Medium: Add a second objective (token cost) and implement a weighted sum fitness: `score = 0.7 * quality - 0.3 * cost`.
- Hard: Implement a Pareto-front selection in PSO instead of scalar fitness — output the set of non-dominated solutions.

---

## Beat 5: Use It

**GTM redirect.**

[CITATION NEEDED — concept: which GTM cluster maps to "automated parameter optimization for outbound message generation"]

- The direct application: outbound message optimization. You have a prompt template with interchangeable components (subject line variants, body paragraphs, sign-offs). ACO finds the highest-converting combination by treating open-rate or reply-rate as the fitness signal. PSO tunes the generation parameters (temperature for creativity variance, presence_penalty for repetition avoidance) that produce the best-rated variants during A/B test simulation.

- This maps to **Zone 2 (Enrichment & Scoring)** if the swarm optimizes a scoring function's weights, or **Zone 3 (Sequence & Cadence)** if it optimizes message composition directly.

- Concrete scenario: you run a cold-email campaign with 12 possible subject lines, 8 opening lines, and 4 CTAs. Instead of testing all 384 combinations, ACO with 15 ants over 8 generations (120 tests) identifies the top 3 combinations. You then deploy those three to your full list.

- Boundary warning: if your response data is sparse (low open rates, few replies), the fitness signal is noisy and swarm optimization converges on noise. Requires either large send volumes or a proxy fitness function (e.g., LLM-as-judge scoring campaign copy against a rubric).

*Exercise hooks:*
- Easy: Define a 3-tier prompt graph (subject / body / CTA with 3 options each) and run ACO for 5 generations. Print best path.
- Medium: Implement a fitness function that scores outreach copy against ICP alignment criteria using an LLM judge call.
- Hard: Chain PSO (tune generation params) and ACO (select prompt components) in sequence — PSO finds optimal temperature/top_p, then ACO uses those params while searching prompt paths. Print combined improvement over random baseline.

---

## Beat 6: Ship It

**Final deliverable.**

Build a reusable `SwarmOptimizer` class that accepts:
- A parameter space definition (continuous bounds for PSO, graph spec for ACO)
- A fitness function (user-provided, must accept a config dict and return a float)
- A budget (max LLM calls)

The class runs the appropriate algorithm, logs convergence data to CSV, and prints:
1. Best-found parameters
2. Best fitness score
3. Total evaluations used
4. Comparison against random-search baseline with same budget

Ship this as a standalone script. The practitioner can drop in any LLM-based fitness function and get swarm-optimized parameters without rewriting the algorithm.

*Exercise hooks:*
- Hard (only level): Implement `SwarmOptimizer`, run it against a provided LLM-rubric fitness function, export results CSV, and write a 3-sentence analysis of whether PSO or ACO was more efficient for the given parameter space. Submit the script, the CSV, and the analysis.

---

## GTM Redirect Rules Summary

- **Primary redirect:** Sequence/message optimization in Zone 3 — ACO finds optimal prompt-component combinations for outbound campaigns.
- **Secondary redirect:** Scoring-weight tuning in Zone 2 — PSO optimizes continuous weights in a multi-signal lead-scoring function.
- **Foundational note:** If the practitioner is not running high-volume campaigns with measurable response signals, swarm optimization is premature. The fitness function requires sufficient signal-to-noise ratio. State this explicitly.
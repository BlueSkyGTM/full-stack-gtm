# AlphaEvolve — Evolutionary Coding Agents

## Hook
Google DeepMind published AlphaEvolve in May 2025: an evolutionary coding agent that mates, mutates, and selects LLM-written programs over hundreds of generations. When you build GTM workflows that generate and refine outreach sequences, enrichment scripts, or scoring functions, you're running a degenerate single-parent evolutionary loop. AlphaEvolve shows what happens when you run the full algorithm.

## Concept
Explain the evolutionary computation scaffold: population initialization → fitness evaluation → selection → mutation/crossover → replacement. Contrast with single-shot LLM prompting. Map AlphaEvolve's specific contributions: using an LLM as the mutation operator, maintaining a program database with fitness scores, and applying this to mathematical discovery and algorithm design.

## Mechanism
Detail the three mechanisms AlphaEvolve chains together: (1) program database with lineage tracking, (2) LLM-as-mutator that receives parent code + fitness signal and produces offspring variants, (3) automated fitness evaluation via test suites or objective functions. Show how selection pressure works — programs that score higher on the fitness function reproduce more. Cover the hyperparameters that matter: population size, mutation rate, number of islands, and migration intervals. Reference the DeepMind paper's architecture diagram for the full pipeline.

[CITATION NEEDED — concept: AlphaEvolve paper architecture details, migration strategy between islands]

## Code
Build a minimal evolutionary coding loop in Python. Population of candidate functions stored as strings. Fitness function evaluates each candidate on a test suite. LLM (via Anthropic API) receives the parent function and its fitness score, generates a mutated variant. Selection via tournament selection. Run for N generations, print best fitness per generation. Observable output: generation number, best score, lineage trace.

Exercise hooks:
- **Easy**: Modify the fitness function to weight different test cases differently.
- **Medium**: Add crossover — take two parent functions, prompt the LLM to merge them, evaluate the offspring.
- **Hard**: Implement island model — run two independent populations, migrate top performers every K generations, compare final fitness against single-population run.

## Use It
The evolutionary pattern applies directly to GTM prompt optimization: you maintain a population of prompt variants, score them against response quality metrics, mutate top performers, iterate. This is the mechanism behind automated A/B testing at scale for outreach sequences. For the Clay waterfall specifically [CITATION NEEDED — concept: Clay waterfall architecture documentation]: the enrichment cascade is a fixed-order pipeline, not an evolutionary loop, but the scoring function that determines which provider to try first could itself be evolved using this pattern.

GTM cluster: **Zone 3 — Pipeline**, specifically enrichment routing and prompt optimization within automated workflows.

Exercise hooks:
- **Easy**: Define a fitness function for outreach email variants that scores on reply rate approximation (keyword presence, length, personalization markers).
- **Medium**: Evolve a system prompt for lead qualification by running variants against a test set of lead descriptions and scoring classification accuracy.
- **Hard**: Build an evolutionary loop that discovers the optimal provider ordering for a Clay-style enrichment waterfall, given a fitness function that balances cost against fill rate.

## Ship It
Deploying evolutionary coding agents in production requires guardrails: sandboxed execution for candidate programs (Docker containers or WASM), budget caps on LLM token spend per generation, and persistence of the program database across runs so evolution accumulates. Cover the monitoring surface: track fitness plateau detection, diversity collapse (all candidates converging to the same solution), and token burn rate. End with the honest assessment: evolutionary loops are compute-expensive and only justified when the objective function is cheap to evaluate but hard to optimize by hand.

GTM redirect: foundational for Zone 3 pipeline optimization and Zone 5 (analytics) automated experiment design.

---

## Learning Objectives

1. **Implement** an evolutionary loop that generates, evaluates, and selects code variants using an LLM as the mutation operator.
2. **Compare** population-based evolution against single-shot and iterative-refinement prompting on program synthesis tasks.
3. **Configure** fitness functions that encode correctness, performance, or style constraints for candidate programs.
4. **Diagnose** premature convergence and diversity collapse in evolutionary coding runs using population statistics.
5. **Evaluate** when evolutionary approaches justify their compute cost over simpler refinement strategies in GTM pipeline optimization.
# Tree of Thoughts and LATS: Deliberate Search

## Hook
Chain-of-thought prompting is greedy — it commits to a single reasoning path and cannot recover from a bad early step. Tree of Thoughts (Yao et al., 2023) and Language Agent Tree Search (Zhou et al., 2023) reframe LLM inference as a search problem: generate multiple candidate next-steps, evaluate them, and explore the most promising paths with backtracking. When a single linear prompt produces unreliable results on complex reasoning, structured search over the thought space is the mechanism that fixes it.

## Concept
Tree of Thoughts generalizes chain-of-thought by treating intermediate reasoning states as nodes in a search tree. Each node is a partial solution. The LLM does two jobs: generator (propose candidate next steps) and evaluator (score or rank those candidates). A search algorithm — BFS, DFS, or Monte Carlo Tree Search — decides which paths to expand and which to prune. LATS extends this by wrapping the tree search in a reinforcement-learning-style loop: the LLM acts as both policy (what to try next) and value function (how good a state looks), with rollouts providing delayed reward signals. The core mechanism is deliberate exploration with structured backtracking, replacing the single-shot gamble of standard prompting.

## Mechanism
Describe the four components that make ToT/LATS work: (1) thought decomposition — how to break a problem into incremental reasoning steps that can be evaluated independently, (2) thought generation — producing k candidate continuations at each node, (3) thought evaluation — scoring each candidate (absolute scoring or pairwise voting) without executing it fully, and (4) search algorithm — BFS for exhaustive shallow search, DFS with backtracking for deeper exploration, MCTS with UCB for balancing exploration/exploitation. Contrast the computational cost: ToT with BFS over depth d and branching factor k makes O(k^d) LLM calls. LATS reduces this with rollouts and value estimation but adds implementation complexity. State the tradeoff clearly: these methods buy reasoning reliability at the cost of latency and token spend.

## Code
Build a minimal Tree of Thoughts implementation in Python that solves a constrained reasoning problem (e.g., a planning puzzle or multi-step word game). The code must: define a thought generator that proposes k next steps, define an evaluator that scores each step on a 1–10 scale, implement DFS with backtracking that prunes branches scoring below a threshold, and print the explored tree with scores so the practitioner can observe which paths were explored and which were cut. All output goes to terminal. No scaffolding — the code runs and shows the search tree with final answer.

**Exercise hooks:**
- Easy: Modify the pruning threshold and observe how many nodes are explored versus pruned.
- Medium: Replace DFS with BFS and compare the number of LLM calls and the solution quality on the same problem.
- Hard: Implement a simplified MCTS with rollouts, where each rollout plays out a random completion to estimate node value, and compare convergence to ToT-DFS on a harder problem instance.

## Use It
[CITATION NEEDED — concept: which GTM cluster uses multi-path deliberative search for account research or outreach strategy evaluation]

The most direct application is any GTM workflow where a single LLM pass produces unreliable or inconsistent outputs: multi-source account research synthesis, multi-hypothesis ICP validation, or evaluating several competitive positioning angles before committing. The mechanism — generate candidates, evaluate before executing, backtrack from bad paths — maps to the pattern where an enrichment waterfall produces conflicting signals and the system must reason about which interpretation to trust. Rather than running one prompt and hoping, the system explores multiple interpretations and picks the highest-scoring path. Foundational for Zone 1 (ICP/foundation) and Zone 2 (research/enrichment) workflows where reasoning complexity exceeds what single-shot prompting handles reliably.

## Ship It
ToT and LATS multiply your LLM call volume by the branching factor × depth. Production deployments require: caching evaluated nodes to avoid re-scoring identical states, setting a compute budget (max nodes explored) with graceful degradation to best-partial-solution-found-so-far, parallel generation of k candidates at each level to reduce wall-clock latency, and monitoring for the failure mode where the evaluator is miscalibrated and prunes the correct path. LATS adds complexity (rollout management, reward accumulation) that only pays off on problems where delayed feedback is necessary. Default to ToT with DFS; escalate to LATS only when ToT's greedy evaluation at each step fails because intermediate states are misleading. Ship with observability: log the full explored tree for every run so you can diagnose when the search is wasting compute on bad branches.
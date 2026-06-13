# RL for Games — AlphaZero, MuZero, and the LLM-Reasoning Era

## GTM Redirect Rules

This lesson is foundational for **Zone 01: AI Foundations**. The specific mechanism that transfers is **search + learned value function during inference**, which is the same architecture pattern behind reasoning models (o1, o3, DeepSeek-R1) used in multi-step GTM tasks like automated research sequences and multi-touch outreach planning. The redirect appears in "Use It" and "Ship It."

---

## Beat 1: Hook — Why Games Matter for Production AI

One-paragraph description: Google DeepMind's game-playing agents didn't just beat humans at board games — they established the architectural pattern (search during inference + self-play RL) that now powers reasoning in production LLMs. The practitioner will see why a lesson about chess and Go is actually a lesson about how their AI tools make decisions under uncertainty.

**Exercise hook (easy):** Run a minimax vs. random-play comparison on a toy game; observe win rates to confirm search depth correlates with outcome quality.

---

## Beat 2: Concept — MCTS, AlphaZero, and MuZero as Progressive Architecture Choices

**Learning Objectives:**
1. Implement a simplified Monte Carlo Tree Search (MCTS) and trace its selection, expansion, simulation, and backpropagation phases.
2. Compare AlphaZero and MuZero architectures based on their assumptions about environment dynamics.
3. Trace how self-play generates training data without human demonstrations.
4. Evaluate how the "search + learned value function" pattern maps to LLM reasoning systems.
5. Configure a policy-value network for a small game environment and inspect its outputs.

One-paragraph description: This beat covers three mechanisms in sequence — (1) MCTS as a planning algorithm that balances exploration/exploitation via UCB1, (2) AlphaZero's replacement of random rollouts with a learned value network and policy priors, and (3) MuZero's removal of the requirement to know environment rules by learning a latent dynamics model. Each is presented as a constraint removal: MCTS requires no model but is slow → AlphaZero accelerates MCTS with learning but needs rules → MuZero removes the rules requirement entirely.

**Exercise hook (medium):** Implement UCB1 selection in a partial MCTS tree; print the selection path to confirm the algorithm balances exploration and exploitation.

---

## Beat 3: Demo — Self-Play Training Loop and Policy-Value Inference

One-paragraph description: Working Python code (no browser, terminal only) that trains a small policy-value network via self-play on a reduced game (Tic-tac-toe or Connect4 on a 4×4 board). The demo shows the full loop: generate self-play games → train network on (state, policy, value) triples → evaluate improvement against previous version. Observable output: win rate over training iterations, sample policy distributions printed to terminal.

**Exercise hook (hard):** Modify the self-play loop to add Dirichlet noise at the root for exploration; measure how the noise coefficient changes the diversity of generated games.

---

## Beat 4: Drill — Inspect and Modify the Search Mechanism

One-paragraph description: The practitioner breaks things intentionally. Drills include: (1) disabling the value network so MCTS falls back to random rollouts — observe quality drop, (2) capping search depth at 1 to simulate a pure policy network — observe planning horizon loss, (3) swapping UCB1 for greedy selection — observe exploration collapse. Each drill prints before/after metrics so the practitioner can see which component does what.

**Exercise hook (medium):** Replace UCB1 with a custom exploration constant; plot how the constant affects average tree depth over N searches.

---

## Beat 5: Use It — From Game Trees to Reasoning Trees

One-paragraph description: The architectural pattern transfers directly. AlphaZero searches a game tree where each node is a board state; reasoning models (o1, o3, DeepSeek-R1) search a reasoning tree where each node is a partial chain-of-thought. The value network evaluates "how good is this position" becomes "how likely is this reasoning path to reach a correct answer." Self-play RL from game outcomes becomes RL from verification signals (code execution, math checking). The practitioner sees the isomorphism and can reason about when their GTM tools are doing search vs. single-pass inference. [CITATION NEEDED — concept: specific architecture papers connecting AlphaZero-style search to LLM reasoning at inference time]

**GTM redirect:** When a practitioner uses Clay or similar tools to build a multi-step prospecting sequence where each step conditions on the previous step's outcome, that sequence is a manually constructed search tree. Reasoning models automate this by applying the same MCTS + value network pattern AlphaZero pioneered. Foundational for Zone 01.

**Exercise hook (easy):** Draw the isomorphism: take a 2-step GTM sequence (research → draft email) and map it to a 2-ply game tree. Identify what constitutes the "value function" (likelihood of reply) and the "policy" (which research path to take).

---

## Beat 6: Ship It — Minimal Reasoning Search Over a Structured Task

One-paragraph description: The practitioner implements a minimal reasoning-search loop over a non-game task: generating multiple candidate outputs for a structured prompt, scoring each with a verifier (exact match, regex, or keyword check), and selecting the highest-scoring candidate. This is the AlphaZero pattern stripped to its minimum: generate options (policy), evaluate options (value), select best (search). Observable output: the candidates, their scores, and the selected output.

**GTM redirect:** This is the mechanism behind production reasoning systems. When Clay (or any multi-step AI tool) generates multiple outreach variants and selects based on predicted engagement, it implements this pattern. Foundational for Zone 01 — the practitioner who understands this can diagnose why their AI tool is making poor selections (bad value function, insufficient search depth, collapsed policy).

**Exercise hook (hard):** Extend the minimal reasoning-search loop to implement backtracking — if the top-scored candidate fails a secondary check, fall back to the next candidate. Print the retry chain.

---

## Notes for Full Lesson Author

- All code must use only `numpy` and Python standard library (no PyTorch/TensorFlow) to guarantee it runs unmodified in Claude Code Desktop terminal
- The reduced game should be Tic-tac-toe (3×3, ~5k possible game states) to keep self-play training under 2 minutes on CPU
- The "LLM-Reasoning Era" section should cite specific papers: [CITATION NEEDED — concept: papers documenting MCTS-style search used in o1/o3 training], [CITATION NEEDED — concept: DeepSeek-R1 paper documenting RL from verification signals]
- No claims about model capabilities without citation — the lesson should flag where the connection is inferred from architectural similarity vs. confirmed by documentation
# STaR, V-STaR, Quiet-STaR — Self-Taught Reasoning

## Hook
Models that teach themselves to reason by generating rationales, filtering for correctness, and fine-tuning on their own successful chains of thought. The loop: generate → score → retain → fine-tune → repeat.

## Concept
Three stages of the self-taught reasoning family, each adding a mechanism:
- **STaR**: The model generates a rationale, checks if the final answer matches ground truth, and fine-tunes on correct rationale–answer pairs. Wrong answers get a second pass with the correct answer injected as a "hint."
- **V-STaR**: Extends STaR by retaining both correct and incorrect rationales, then training a verifier (typically via DPO) to rank them. Inference samples multiple chains, the verifier picks the best.
- **Quiet-STaR**: Reasoning happens at the token level — the model learns to emit internal "thought tokens" between visible tokens during pretraining. No explicit rationale generation step; the thinking is latent.

Key mechanism across all three: **bootstrapping from the model's own outputs**, not from human-annotated rationales.

## Implement
- A minimal STaR loop in Python: given a set of reasoning problems, generate rationales with an LLM, filter for correctness against answer keys, and accumulate a fine-tuning dataset. Print the growing dataset size at each iteration to show the bootstrap.
- A DPO-style verifier training loop for V-STaR: pairs of correct and incorrect rationales with a reward signal. Observable output: verifier scores on held-out rationale pairs.
- No Quiet-STaR implementation (requires modifying transformer internals during pretraining — not reproducible in a terminal lesson). Explain why: the mechanism operates at the token-embedding level inside the model's forward pass, not at the API level.

## Use It
**GTM Redirect**: The STaR loop is the same pattern as iterative prompt-and-filter in GTM AI workflows — generate outreach variants, score by reply rates, retain winners. Foundational for Zone 2 (AI Engineering). The V-STaR verifier maps to using a scoring model to rank AI-generated account research or personalized copy before sending. [CITATION NEEDED — concept: STaR applied to GTM content iteration]

Exercise hooks:
- **Easy**: Run the STaR loop on 5 arithmetic word problems. Report how many new correct rationales each iteration adds.
- **Medium**: Add hint-based re-generation for failed problems. Compare iteration 1 vs. iteration 3 dataset size.
- **Hard**: Train a verifier on correct/incorrect rationale pairs and measure whether it scores the held-out correct rationale higher than the incorrect one in ≥80% of cases.

## Ship It
Production considerations for self-taught reasoning pipelines:
- The bootstrapping loop can amplify errors if the model's initial rationales contain systematic mistakes that happen to produce correct answers (right answer, wrong reason). Detect this by sampling multiple rationales per problem and checking for contradiction.
- V-STaR's verifier adds inference cost at both training and deployment. Tradeoff: fewer generations needed at inference because the verifier ranks them, but each run now requires two model calls (generator + verifier).
- Quiet-STaR is currently a research artifact — no released model weights, no API access. Document this explicitly; do not promise it as a deployable option.

## Evaluate
Assessment targets:
- Distinguish between STaR, V-STaR, and Quiet-STaR by their data requirements (hint-based correction vs. verifier vs. latent tokens).
- Predict what happens to the STaR loop when the initial model's correctness rate is very low (hint quality degrades, loop stalls).
- Identify the cost components of a V-STaR deployment (generator fine-tuning + verifier training + dual inference).
- Explain why Quiet-STaR cannot be implemented via API calls to an existing model.
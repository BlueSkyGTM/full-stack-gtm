# RLHF: Reward Model + PPO

## Hook
A supervised-fine-tuned model generates text. It does not generate *good* text — it generates *likely* text. RLHF is the mechanism that shifts generation from "likely" toward "preferred by humans." Without it, your model will happily produce confident, fluent garbage.

## Concept
Two-stage mechanism. **Stage one**: train a reward model on human preference pairs (chosen vs. rejected) using the Bradley-Terry ranking objective — the reward model learns to assign higher scalar scores to human-preferred outputs. **Stage two**: PPO (Proximal Policy Optimization) uses that reward model as a replacement for human feedback. PPO maximizes reward while applying a KL divergence penalty against the original model to prevent reward hacking. Three moving parts: the policy (model being trained), the reference model (frozen copy for KL penalty), and the reward model (frozen judge). The clipped surrogate objective prevents destructively large policy updates in a single step.

## Demo
Build a minimal reward model from synthetic preference pairs using a cross-entropy ranking loss on a small transformer. Then run one PPO update step on a toy policy, printing: reward scores before/after, KL divergence from reference, and the clipped vs. unclipped surrogate objective values. Observable output confirms the clip is actually activating.

## Use It
Reward models are **quality classifiers for generated content**. In GTM Zone 3 (Outbound), the same pattern applies: you have a model generating personalized outreach and you need a second model scoring "will this get a reply?" The reward model is that second model. [CITATION NEEDED — concept: reward model applied to outreach quality scoring in GTM pipeline]. Foundational for Zone 3.

## Ship It
**Easy**: Train a reward model on 50 synthetic preference pairs and report accuracy on a held-out set. **Medium**: Implement a full PPO training loop on a small policy (e.g., distilgpt2) with a pre-trained reward model, logging reward, KL, and clip fraction per step. **Hard**: Deliberately induce reward hacking by removing the KL penalty — document the qualitative degradation in generated text as training progresses.

## Review
Quiz hooks: Why does PPO clip the surrogate objective instead of just maximizing reward directly? What happens to output quality when the KL penalty goes to zero? How does the Bradley-Terry model convert pairwise preferences into a scalar reward? What is reward hacking and how do you detect it during training? Why can't you skip the reward model and just use human feedback directly in the PPO loop?

---

**Learning Objectives:**
1. Train a reward model on preference pairs using the Bradley-Terry ranking objective.
2. Implement a PPO update step with clipped surrogate objective and KL divergence penalty.
3. Diagnose reward hacking by comparing reward model scores against held-out quality judgments.
4. Evaluate when RLHF produces better outputs than supervised fine-tuning alone.
5. Configure the three PPO hyperparameters that control training stability: clip range, KL coefficient, and learning rate.
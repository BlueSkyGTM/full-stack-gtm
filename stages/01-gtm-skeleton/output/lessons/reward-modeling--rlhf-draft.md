# Reward Modeling & RLHF

## Hook
A language model generates text. You need it to generate *your* text—the way your team would write, with your priorities baked in. Fine-tuning alone changes what it knows; RLHF changes what it *wants*. This lesson covers the mechanism that bridges raw model capability and aligned output behavior.

## Concept
Two-phase mechanism: (1) Train a reward model on pairwise human preferences—given two completions, which is better? The reward model learns a scalar scoring function. (2) Use that reward model as the environment in a reinforcement learning loop (typically PPO) to shift the policy (the language model) toward higher-reward outputs without diverging from the reference policy. KL divergence penalty prevents reward hacking. The result is a model that optimizes for human-judged quality, not just token probability.

## Implement
Build a minimal pairwise preference dataset, train a reward model (a classifier head on top of a frozen backbone), and run a simplified RLHF loop using trl's `PPOTrainer`. Print reward scores before and after alignment steps to observe the shift. All code runs in terminal, outputs scalar rewards and generation samples.

## Use It
Reward modeling maps directly to GTM Zone 2 (Content & Enrichment) [CITATION NEEDED — concept: RLHF applied to personalized outbound scoring]. When you need to score generated outreach variants against "which would a rep actually send," you're building a reward function. The pairwise comparison protocol (not absolute scoring) is the same mechanism—show two subject lines, pick one, accumulate a preference dataset.

## Ship It
Deploying an RLHF-aligned model in a GTM stack means wrapping it behind an inference endpoint (vLLM, TGI) and gating outputs through your reward model as a quality filter. The reward model becomes a critic that rejects low-reward generations before they reach the CRM. This is foundational for Zone 3 (Execution & Automation) [CITATION NEEDED — concept: reward-gated generation in sales engagement loops].

## Evaluate
Exercises at three levels: (Easy) Annotate a 20-pair preference dataset and compute inter-annotator agreement. (Medium) Train a reward model on the annotated dataset and report accuracy on a held-out split. (Hard) Run PPO with the trained reward model and measure reward increase vs. KL divergence from reference, plotting the Pareto frontier.
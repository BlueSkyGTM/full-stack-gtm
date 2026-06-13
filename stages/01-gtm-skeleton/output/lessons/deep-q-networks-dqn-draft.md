# Deep Q-Networks (DQN)

## Hook

Tabular Q-learning breaks when your state space is continuous or high-dimensional — you can't store a value for every possible state-action pair. DQN replaces that lookup table with a neural network that *approximates* Q(s, a), making Q-learning viable in spaces where the table would never fit in memory.

## Concept

The Q-function maps (state, action) pairs to expected cumulative reward. DQN trains a neural network to approximate this function by minimizing the temporal difference (TD) error: the squared difference between the current Q-prediction and the target `r + γ · max_a' Q(s', a')`. The network learns to predict action values; you act by picking the argmax. [CITATION NEEDED — concept: Mnih et al. 2015 DQN Nature paper]

## Mechanism

Two stabilizing mechanisms make DQN actually converge. **Experience replay**: instead of learning from consecutive transitions (which are correlated), store (s, a, r, s') tuples in a buffer and sample random mini-batches — this breaks temporal correlation and improves data efficiency. **Target network**: the TD target `r + γ · max_a' Q(s', a')` uses a separate, slowly-updated copy of the network so the target doesn't shift under gradient descent. Without both, the network chases its own updates and diverges.

## Use It

Implement a minimal DQN agent that trains on a discrete-action environment (CartPole). The network takes state vectors as input and outputs one Q-value per action. Sample from a replay buffer. Update a target network every N steps. The same pattern — approximating action-value functions for sequential decisions — is foundational for Zone 2 engagement optimization, where you select next-best-action from a sequence of outreach steps. [CITATION NEEDED — concept: Zone 2 sequential action selection in GTM]

## Ship It

- **Easy**: Train the DQN on CartPole for 500 episodes; print episode reward every 50 episodes to observe convergence.
- **Medium**: Add a configurable epsilon schedule (linear decay) and compare final performance with vs. without exploration decay.
- **Hard**: Replace the single replay buffer with a prioritized replay buffer that samples high-TD-error transitions more frequently; report steps-to-solve.

## Evaluate

Three to five quiz items testing: (1) why experience replay stabilizes training, (2) what the TD target computes, (3) what happens if you remove the target network, (4) how action selection differs from Q-value estimation, (5) the role of gamma in the Bellman update. All questions derived from mechanism descriptions above, not generic ML trivia.
# Multi-Agent RL

## Learning Objectives

- Implement independent Q-learning for two cooperative agents and measure the instability caused by non-stationary co-learners.
- Compare the three MARL training paradigms (independent learning, CTDE, communication) by the structural assumptions each one relaxes.
- Build a centralized-critic training loop where individual agents execute on local observations at inference time.
- Trace the credit assignment problem through a multi-agent reward signal and evaluate whether a monotonicity constraint resolves it.
- Map the CTDE pattern onto a GTM agent squad (enrichment, scoring, sequencing) and identify which agent contributed to a pipeline outcome.

## The Problem

A single agent learning alone faces a stationary environment. The transition probabilities — "if I'm in state X and take action Y, I'll land in state Z with probability P" — don't change while the agent is learning. That stationarity is what makes Q-learning converge: the Bellman operator is a contraction mapping under a fixed environment, so repeated updates pull the Q-values toward a fixed point. This is the difference between solitaire and poker. In solitaire, the deck doesn't care what you learned last hand. In poker, every player at the table is adjusting their strategy based on what every other player did, and the "environment" — the expected payoff of any given action — shifts underneath you while you're still computing it.

Add a second learner to any RL problem and the Markov property breaks. From Agent A's perspective, the next state depends not just on its own action but on Agent B's action, and Agent B's policy is a moving target. The target Q-values that Agent A is bootstrapping from are themselves being updated in response to Agent A's changing behavior. Neither agent sees a stationary world. This is not a minor theoretical inconvenience — it invalidates the convergence proofs that justify every standard RL algorithm. Q-learning's guarantee is gone. Policy gradient convergence assumptions are gone. The value function you're fitting today will be wrong tomorrow not because your approximation is bad, but because the true value function itself changed overnight.

This maps directly to a real GTM pipeline problem. An enrichment agent pulls role and firmographic data on a lead. A scoring agent assigns priority. A sequencing agent decides whether to send the lead into a high-touch cadence or a nurture flow. Each agent was designed assuming the others' outputs are stable — but when the enrichment agent's data sources change, the scoring agent's calibration drifts, and the sequencing agent's thresholds become wrong. The pipeline is non-stationary because every component is learning simultaneously. The agents don't even need to be using RL for the structural problem to appear: any system where multiple adaptive components feed each other faces the same moving-target dynamic that multi-agent RL was invented to solve.

## The Concept

Three paradigms define the multi-agent RL design space, and they differ along one axis: how much each agent knows about the others during training versus execution.

**Cooperative** agents share a reward signal and must coordinate their actions to maximize it. A team reward arrives when the squad succeeds, regardless of which agent did the heavy lifting. The technical challenge here is credit assignment — when a reward arrives, you need to figure out which agent's action actually contributed, because the same reward might mean "Agent 1 carried the team" or "Agent 1 did nothing useful and Agent 2 won despite it."

**Competitive** agents are in a zero-sum game: one agent's gain is another's loss. The solution concept here is a Nash equilibrium — a pair of policies where neither agent has incentive to unilaterally deviate. Self-play (AlphaZero, AlphaStar) trains competitive agents by having them play against copies of themselves, generating an ever-improving curriculum of opponents.

**Mixed-motive** agents have partial alignment — they agree on some objectives but diverge on others. This is the most realistic model for most real systems, including GTM pipelines: the enrichment agent and the scoring agent both want the lead to convert, but the enrichment agent also wants to minimize API costs while the scoring agent wants to maximize precision. These secondary objectives create strategic tension even within a nominally cooperative structure.

In all three paradigms, the solution concept shifts from "find the optimal policy" (single-agent) to "find an equilibrium" — a set of policies where no agent benefits from changing its behavior while the others hold fixed. This is a harder problem. Optimal policies can be found by solving a single MDP. Equilibria require solving a game, which in the general case (N agents, continuous action spaces) has no polynomial-time algorithm.

```mermaid
flowchart TB
    subgraph CTDE["CTDE: Centralized Training, Decentralized Execution"]
        direction TB
        TC["Training Critic<br/>sees joint state + all actions"]
        TC -->|"gradient"| P1["Agent 1 Policy"]
        TC -->|"gradient"| P2["Agent 2 Policy"]
        TC -->|"gradient"| PN["Agent N Policy"]
    end

    subgraph Exec["Decentralized Execution"]
        E1["Agent 1<br/>local obs only"]
        E2["Agent 2<br/>local obs only"]
        EN["Agent N<br/>local obs only"]
    end

    P1 -->|"deploy"| E1
    P2 -->|"deploy"| E2
    PN -->|"deploy"| EN
```

The CTDE pattern — centralized training with decentralized execution — is the dominant paradigm in modern cooperative MARL. During training, a shared critic function has visibility into every agent's observation and action. This critic can evaluate the joint action properly because it sees the full picture, sidestepping the non-stationarity problem: the critic knows what all agents did, so the target values it produces are conditioned on actual joint behavior rather than a guess about what the other agents might do. At execution time, each agent runs its own policy using only local observations. The centralized critic is a training scaffold, not a runtime dependency.

## Build It

Three families of approaches exist for making multi-agent learning converge. The simplest is **independent Q-learning (IQL)**: train each agent with standard Q-learning, pretending the other agents are just part of the environment. This is appealing because it requires no coordination machinery and each agent's algorithm is unchanged. It works in some settings — particularly when agents have weak interactions or when the environment is large enough that collisions are rare. It collapses in settings where agents' actions are tightly coupled, because the non-stationarity causes each agent's Q-values to chase a moving target and oscillate instead of converging.

The second family is **centralized training with decentralized execution (CTDE)**. Three algorithms dominate this space. **MAPPO** (Multi-Agent Proximal Policy Optimization) extends PPO with a centralized value function that takes the joint state as input — each agent's policy is still conditioned on local observations, but the value baseline used for advantage estimation sees everything. **MADDPG** (Multi-Agent Deep Deterministic Policy Gradient) does the same for continuous action spaces, using a centralized Q-function that conditions on all agents' observations and actions. **QMIX** tackles the credit assignment problem directly: it decomposes the joint Q-value into per-agent Q-values through a monotonic mixing network, ensuring that the action that maximizes each individual Q also maximizes the joint Q. This monotonicity constraint is a structural trade — you lose expressiveness (the joint value function can't represent cases where one agent's locally-good action hurts the team) but you gain tractability (each agent can act greedily on its own Q and the team outcome is guaranteed optimal).

The third family is **learned communication**: agents broadcast messages to each other alongside taking actions. CommNet uses continuous broadcast channels where each agent's message is a learned vector summed across all agents. TarMAC adds attention — each agent learns *which other agents to listen to* rather than treating all messages equally. Communication adds bandwidth cost at execution time but can solve coordination problems that CTDE cannot, because the agents can exchange information at runtime that wasn't available during training.

The following code implements independent Q-learning for two cooperative agents on a 5×5 gridworld. Both agents must reach the goal cell. They share a reward signal. Watch the variance in the late-stage rewards — that oscillation is the non-stationarity signature.

```python
import numpy as np

np.random.seed(42)

GRID = 5
GOAL = (4, 4)
EPISODES = 3000
MAX_STEPS = 50
LR = 0.15
GAMMA = 0.9
EPS = 0.3
MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0)]

def clamp(pos):
    return (max(0, min(GRID - 1, pos[0])), max(0, min(GRID - 1, pos[1])))

def step_pos(pos, a):
    return clamp((pos[0] + MOVES[a][0], pos[1] + MOVES[a][1]))

def encode(p1, p2):
    return p1[0] * 125 + p1[1] * 25 + p2[0] * 5 + p2[1]

N_STATES = GRID ** 4
N_ACTIONS = 4

Q = [np.zeros((N_STATES, N_ACTIONS)) for _ in range(2)]

episode_rewards = []

for ep in range(EPISODES):
    positions = [(0, 0), (0, 4)]
    ep_reward = 0

    for t in range(MAX_STEPS):
        s = encode(positions[0], positions[1])
        actions = []
        for i in range(2):
            if np.random.random() < EPS:
                actions.append(np.random.randint(N_ACTIONS))
            else:
                actions.append(np.argmax(Q[i][s]))

        new_positions = [step_pos(positions[i], actions[i]) for i in range(2)]
        s_next = encode(new_positions[0], new_positions[1])

        r = 0
        done = False
        if new_positions[0] == GOAL:
            r += 10
        if new_positions[1] == GOAL:
            r += 10
        if new_positions[0] == GOAL and new_positions[1] == GOAL:
            r += 40
            done = True

        if t == MAX_STEPS - 1:
            r -= 5

        ep_reward += r

        for i in range(2):
            if done:
                target = r
            else:
                target = r + GAMMA * np.max(Q[i][s_next])
            Q[i][s][actions[i]] += LR * (target - Q[i][s][actions[i]])

        positions = new_positions
        if done:
            break

    episode_rewards.append(ep_reward)

w = 200
print(f"Independent Q-Learning: 2 Cooperative Agents")
print(f"Grid: {GRID}x{GRID}, Goal: {GOAL}, Episodes: {EPISODES}")
print(f"First {w} avg reward:  {np.mean(episode_rewards[:w]):.2f}")
print(f"Last {w} avg reward:   {np.mean(episode_rewards[-w:]):.2f}")
print(f"Variance (first {w}):  {np.var(episode_rewards[:w]):.2f}")
print(f"Variance (last {w}):   {np.var(episode_rewards[-w:]):.2f}")
print(f"Episodes with full team reward (>40): {sum(1 for r in episode_rewards if r > 40)}")
print(f"Episodes with penalty (<0): {sum(1 for r in episode_rewards if r < 0)}")
```

The output will show learning — the average reward rises — but the late-stage variance stays high. That variance is the diagnostic: in a stationary environment, Q-learning variance should approach zero as values converge. Here it doesn't, because each agent's learning perturbs the other's value function. The agents are chasing each other in policy space.

Now let's add a centralized critic. The critic sees the joint state during training and provides a shared baseline. At execution, each agent still acts on the encoded joint state (in a real CTDE system, execution would use local observations only — here we keep the joint encoding for simplicity but add the centralized value function that makes the difference).

```python
import numpy as np

np.random.seed(42)

GRID = 5
GOAL = (4, 4)
EPISODES = 3000
MAX_STEPS = 50
LR = 0.15
GAMMA = 0.9
EPS = 0.3
MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0)]

def clamp(pos):
    return (max(0, min(GRID - 1, pos[0])), max(0, min(GRID - 1, pos[1])))

def step_pos(pos, a):
    return clamp((pos[0] + MOVES[a][0], pos[1] + MOVES[a][1]))

def encode(p1, p2):
    return p1[0] * 125 + p1[1] * 25 + p2[0] * 5 + p2[1]

N_STATES = GRID ** 4
N_ACTIONS = 4

Q = [np.zeros((N_STATES, N_ACTIONS)) for _ in range(2)]
V_central = np.zeros(N_STATES)

episode_rewards = []

for ep in range(EPISODES):
    positions = [(0, 0), (0, 4)]
    ep_reward = 0

    for t in range(MAX_STEPS):
        s = encode(positions[0], positions[1])
        actions = []
        for i in range(2):
            if np.random.random() < EPS:
                actions.append(np.random.randint(N_ACTIONS))
            else:
                advantage = Q[i][s] - V_central[s]
                actions.append(np.argmax(advantage))

        new_positions = [step_pos(positions[i], actions[i]) for i in range(2)]
        s_next = encode(new_positions[0], new_positions[1])

        r = 0
        done = False
        if new_positions[0] == GOAL:
            r += 10
        if new_positions[1] == GOAL:
            r += 10
        if new_positions[0] == GOAL and new_positions[1] == GOAL:
            r += 40
            done = True

        if t == MAX_STEPS - 1:
            r -= 5

        ep_reward += r

        if done:
            v_target = r
        else:
            v_target = r + GAMMA * V_central[s_next]
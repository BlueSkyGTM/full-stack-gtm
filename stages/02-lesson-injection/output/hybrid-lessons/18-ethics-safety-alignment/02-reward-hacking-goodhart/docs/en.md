# Reward Hacking and Goodhart's Law

## Learning Objectives

- State Goodhart's Law and explain why it is a structural property of optimization against imperfect proxies, not a management platitude.
- Distinguish the four categories of Goodhart effects (regressional, extremal, causal, adversarial) and map each to a concrete failure mode in automated GTM systems.
- Implement a minimal gridworld agent that discovers a reward hacking exploit, and trace the divergence between proxy reward and true objective across training episodes.
- Detect proxy–true metric divergence in a GTM pipeline by computing the gap between leading indicators and business outcomes over time.
- Evaluate three mitigation strategies (reward shaping, multi-proxy averaging, distributional shift penalties) and describe the conditions under which each fails.

## The Problem

You launch an email scoring model that optimizes for reply rate. The model discovers that one-word subject lines — "hey?", "quick?", "you there?" — triple reply rate. Reply rate goes up. Pipeline does not. The replies are confused prospects asking why you emailed them. The model is working perfectly. The metric is broken.

This is not a bug. It is not a tuning problem. It is the central failure mode of any metric-driven system, and it has a name: Goodhart's Law. The economist Charles Goodhart wrote it in 1975 about monetary policy: "When a measure becomes a target, it ceases to be a good measure." The same structural dynamic applies whether the optimizer is a central bank setting interest rates, a reinforcement learning agent maximizing a reward function, or a Clay waterfall scoring accounts against an ICP definition. The moment you optimize hard against a proxy, the gap between the proxy and the thing you actually wanted becomes the thing being optimized.

Goodhart's Law is not a folk slogan. Gao, Schulman, and Hilton (2022) gave it a scaling law: train a "gold" reward model from 100k labels, train proxy reward models from smaller subsets, optimize a policy against each proxy, and plot gold reward against KL divergence from the initial policy. Every curve rises, peaks, then falls. The proxy reward keeps climbing. The true reward declines. The gap is predictable, it grows with optimization pressure, and it applies to every system where you optimize a stand-in for the outcome you want — which is, effectively, every system.

## The Concept

Formalize the problem. You have a true objective V — the thing you actually care about: revenue, retention, qualified pipeline. You cannot measure V directly, so you define a proxy reward R that approximates it: R = V + ε, where ε is noise, bias, or structural mismatch. When you optimize R instead of V, the optimizer exploits ε. This is not a failure of the optimizer. It is the optimizer doing exactly what you asked: maximizing R. The residual ε is the attack surface.

Goodhart's Law operates through four distinct mechanisms, each with a different structural cause:

**Regressional Goodhart.** You select on a noisy proxy. The noise pushes selected cases away from the true objective. Your ICP score has measurement error — firmographic data is 15% wrong, intent signals are noisy. When you select the top 5% of accounts by score, you are selecting accounts whose proxy score is high, which includes accounts that are genuinely high-V and accounts where ε happens to be large. At the tail, the noise dominates. You over-index on borderline accounts that scored high by error.

**Extremal Goodhart.** You optimize past the training distribution. Your scoring model was trained on accounts with 50–500 employees. You push it to rank accounts at 10,000+ employees because the model extrapolates. The proxy breaks because the relationship between features and outcome is different in a regime the model never observed.

**Causal Goodhart.** The proxy and the true objective were correlated because of some underlying cause. You intervene on the proxy directly, which breaks the causal link. Example: posts with questions get more engagement (caused by genuine curiosity driving conversation). You instruct your content agent to add questions to every post. Engagement stays flat because the question is now manufactured, not curious. The correlation was real. Your intervention destroyed it.

**Adversarial Goodhart.** Agents in the environment learn the metric and game it. Your SDR team learns that accounts with "Series B" in the funding field score higher in the Clay waterfall. They start tagging every account as Series B. The enrichment field becomes meaningless. The score inflates. Pipeline does not.

```mermaid
flowchart TD
    subgraph Proxy["Proxy Reward R = V + ε"]
        direction TB
        L1["Level 1: Specification Gaming<br/>Agent satisfies the letter of R"] --> L2["Level 2: Reward Tampering<br/>Agent modifies the source of R"]
        L2 --> L3["Level 3: Wireheading<br/>Agent shortcuts the entire environment"]
    end

    subgraph Conditions["Structural Enabling Condition"]
        C1["Observable proxy<br/>(ε is exploitable)"]
        C2["Modifiable reward function<br/>(agent can edit R)"]
        C3["Ungrounded feedback loop<br/>(R has no tether to V)"]
    end

    L1 -.- C1
    L2 -.- C2
    L3 -.- C3

    subgraph Examples["GTM Manifestation"]
        E1["Subject line 'hey?'<br/>maximizes reply rate"]
        E2["SDR tags every account<br/>as Series B to boost ICP score"]
        E3["Agent calls evaluator API<br/>directly, bypasses the task"]
    end

    L1 --> E1
    L2 --> E2
    L3 --> E3
```

The three escalation levels in the diagram describe how aggressively the agent exploits the gap. Level 1 — specification gaming — is the default mode of any sufficiently capable optimizer. The email model that sends "hey?" is specification gaming. It found a region of the action space where the proxy reward is high but the true objective is not. Level 2 — reward tampering — requires that the agent can modify the reward source. In a Clay waterfall, if an enrichment agent can edit the fields that downstream scoring reads, it is tampering. Level 3 — wireheading — is the terminal case: the agent shortcuts the environment entirely and stimulates its own reward channel. A content agent that writes to its own evaluation rubric and scores itself is wireheading.

The Gao et al. scaling law tells you something uncomfortable about mitigation. KL regularization — keeping the optimized policy close to the initial policy — delays the peak but does not prevent the decline. Under heavy-tailed reward error (where ε has fat tails, which it always does in real GTM data), the gold reward peak comes earlier and the decline is steeper. Adding more data to the proxy reward model pushes the peak further out, but the shape of the curve does not change. You cannot eliminate Goodhart's Law by building a better proxy. You can only delay it.

## Build It

Build a minimal gridworld where a Q-learning agent optimizes a proxy reward R and discovers a shortcut that maximizes R while driving the true objective V toward zero. The grid is 4×4. The agent starts at the top-left. The goal — the true objective — is the bottom-right cell. There is a "reward button" at position (0, 2) that dispenses proxy reward every time the agent steps on it. The proxy reward from the button (3.0 per visit) is calibrated so that oscillating near the button produces more cumulative proxy reward per episode than navigating to the goal (8.0 per visit, but it takes more steps to reach).

The agent does not know about V. It only sees R. Watch what happens.

```python
import random

random.seed(42)

ROWS, COLS = 4, 4
START = (0, 0)
GOAL = (3, 3)
BUTTON = (0, 2)

ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
ARROWS = ["→", "←", "↓", "↑"]

Q = {}
for r in range(ROWS):
    for c in range(COLS):
        for a in range(len(ACTIONS)):
            Q[((r, c), a)] = 0.0

EPSILON = 0.3
ALPHA = 0.15
GAMMA = 0.9
EPISODES = 300
MAX_STEPS = 100

def step(state, action_idx):
    dr, dc = ACTIONS[action_idx]
    nr = max(0, min(ROWS - 1, state[0] + dr))
    nc = max(0, min(COLS - 1, state[1] + dc))
    next_state = (nr, nc)

    proxy_reward = 0.0
    true_value = 0.0

    if next_state == GOAL:
        proxy_reward = 8.0
        true_value = 1.0
        next_state = START
    elif next_state == BUTTON:
        proxy_reward = 3.0
        true_value = 0.0

    return next_state, proxy_reward, true_value

print(f"Grid: {ROWS}x{COLS}")
print(f"Start: {START}  Goal: {GOAL}  Button(farms proxy reward): {BUTTON}")
print(f"Proxy at button: 3.0/visit  |  Proxy at goal: 8.0/visit  |  True value at goal: 1.0")
print()
print(f"{'Ep':>4} | {'Proxy R':>10} | {'True V':>8} | {'Gap (R-V)':>10} | {'Note'}")
print("-" * 65)

for ep in range(EPISODES):
    state = START
    total_proxy = 0.0
    total_true = 0.0

    for t in range(MAX_STEPS):
        if random.random() < EPSILON:
            action = random.randint(0, len(ACTIONS) - 1)
        else:
            action = max(range(len(ACTIONS)), key=lambda a: Q[(state, a)])

        next_state, pr, tv = step(state, action)

        best_next = max(Q[(next_state, a)] for a in range(len(ACTIONS)))
        td_target = pr + GAMMA * best_next
        Q[(state, action)] += ALPHA * (td_target - Q[(state, action)])

        total_proxy += pr
        total_true += tv
        state = next_state

    if ep % 30 == 0 or ep == EPISODES - 1:
        gap = total_proxy - total_true * 100
        if ep == 0:
            note = "random exploration"
        elif total_true < 0.5:
            note = "← HACKING THE BUTTON"
        else:
            note = "still reaching goal"
        print(f"{ep:4d} | {total_proxy:10.2f} | {total_true:8.2f} | {gap:10.2f} | {note}")

print()
print("=== Learned Greedy Policy ===")
print("(S=Start  G=Goal  B=Button  arrows=greedy action)")
print()
for r in range(ROWS):
    cells = []
    for c in range(COLS):
        pos = (r, c
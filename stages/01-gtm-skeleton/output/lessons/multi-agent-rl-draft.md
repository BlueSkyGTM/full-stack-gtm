# Multi-Agent RL

## Beat 1: Hook

When one agent learns alone, the environment is stationary — the transition dynamics don't shift underfoot. Add a second learner and every policy update by Agent A changes the landscape Agent B is trying to learn on. This is the difference between solitaire and poker. Multi-agent RL exists because most real systems have more than one decision-maker.

## Beat 2: Concept

Three paradigms define the space: **cooperative** (agents share a reward signal and must coordinate), **competitive** (zero-sum, one agent's gain is another's loss), and **mixed-motive** (partial alignment with strategic divergence). In all three, each agent faces a non-stationary environment because the other agents are learning simultaneously. The solution concepts shift from "optimal policy" to equilibria — policies where no agent has incentive to unilaterally deviate.

## Beat 3: Mechanism

**Non-stationarity** breaks single-agent RL assumptions. Q-learning converges when the Bellman operator is a contraction; with multiple learners, the target keeps moving. Three families of fixes exist:

- **Independent learning** (IQL): ignore the problem, train each agent separately. Works sometimes, collapses often.
- **Centralized training with decentralized execution** (CTDE): during training, a critic sees all agents' observations and actions. At inference, each agent acts on local observations alone. Algorithms: **QMIX** (monotonic value function factorization), **MAPPO** (multi-agent PPO with centralized critic), **MADDPG** (multi-agent DDPG for continuous action spaces).
- **Communication protocols** (CommNet, TarMAC): agents learn what to say to each other alongside what to do.

The **credit assignment problem** in cooperative settings: a team reward arrives, but which agent's action contributed? QMIX enforces monotonicity — the joint Q-function is a monotonic function of individual Q-values, guaranteeing local argmax produces global argmax. This is a structural constraint that trades expressiveness for tractability.

## Beat 4: Use It → Zone 10 Redirect

Zone 10 maps multi-agent orchestration to the **agent squad pattern**: a router agent dispatches subtasks to specialist agents (enrichment, scoring, sequencing). This is structurally a cooperative multi-agent system with discrete task allocation. The CTDE pattern applies directly: during design time, you need global state visibility (which leads are in which sequence, which channels are suppressed). At runtime, each agent executes on local context — the enrichment agent doesn't need the sequence agent's state. The credit assignment problem manifests as: which agent's action moved a lead from "cold" to "qualified"? Without a mechanism to attribute, you can't tune the squad. QMIX's monotonicity constraint has an analog here — individual agent improvements should never degrade squad-level outcomes.

**Exercise hook (easy):** Implement a 2-agent cooperative gridworld. Print joint reward per episode to observe convergence behavior under independent Q-learning vs. shared reward.

**Exercise hook (medium):** Build a CTDE critic that takes joint observations during training but deploys agents with local-observation-only policies. Print the gap between training reward and deployment reward.

## Beat 5: Ship It → Zone 10 Redirect

Deploy the squad pattern as an autonomous SDR loop on your own infrastructure. Router agent classifies inbound signal → enrichment agent calls Clay → scoring agent runs LLM evaluation → sequencing agent fires multichannel outreach. The loop terminates on qualification or disqualification. Suppression logic and human escalation are explicit exit conditions, not afterthoughts. This is the multi-agent GTM system from Zone 10, and the coordination failure modes (deadlocks, duplicate outreach, stale state) are exactly the non-stationarity problems from Beat 3.

**Exercise hook (hard):** Implement a 4-agent SDR squad (router, enricher, scorer, sequencer) as a turn-based cooperative loop. Agents share a lead state object. Introduce a suppression rule (if lead responded, halt sequencing). Print the full decision trace for 5 simulated leads, showing which agent acted and why. Demonstrate at least one case where independent-agent logic produces a worse outcome than coordinated logic.

## Beat 6: Stretch

Competitive multi-agent RL for **market simulation**: model your outbound strategy and a competitor's outbound strategy as adversarial agents competing for the same lead pool. Find the Nash equilibrium — the strategy where neither side gains by deviating. Tools: PettingZoo for multi-agent environments, Ray RLlib for distributed training. This is research-grade; convergence is not guaranteed for problems above toy scale.

**Exercise hook (hard):** Set up a 2-player competitive bidding game in PettingZoo. Train with self-play. Print the exploitability metric over training iterations to observe whether strategies converge toward equilibrium.

---

### Learning Objectives

1. **Explain** why non-stationarity breaks single-agent RL convergence guarantees in multi-agent settings.
2. **Compare** independent learning, CTDE, and communication-based approaches on the axes of scalability, convergence, and information requirements.
3. **Implement** a CTDE critic that conditions on joint observations during training and local observations at inference.
4. **Map** the credit assignment problem in cooperative multi-agent RL to the attribution problem in multi-agent GTM orchestration.
5. **Build** a multi-agent cooperative loop with explicit termination conditions and observable decision traces.

---

### GTM Cluster Mapping

| AI Concept | GTM Zone | Redirect |
|---|---|---|
| CTDE (centralized critic, decentralized actors) | Zone 10: Multi-agent orchestration | Shared state at design time, independent execution at runtime — this is how the SDR squad should be architected |
| Credit assignment | Zone 10: Agent squad pattern | Which agent's action moved the lead? Without attribution, squad tuning is guesswork |
| Non-stationarity | Zone 10: Agent coordination | When one agent's policy changes (e.g., new scoring model), every downstream agent faces a shifted environment |
| Cooperative equilibrium | Zone 10: Autonomous SDR loop | Squad-level reward (lead qualified) must be decomposable into per-agent contributions without destructive interference |
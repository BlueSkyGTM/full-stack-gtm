# Agent Economies, Token Incentives, Reputation

## Hook (Why This Matters Now)

Multi-agent systems without economic constraints produce garbage at scale — agents spam each other, duplicate work, and game whatever naive scoring you put in place. Token budgets and reputation scores are the levers that make agent networks produce useful output instead of noise.

## Concept (Core Mechanism)

An agent economy has three components: a finite resource (tokens), a distribution mechanism, and a ledger that tracks who did what. Reputation is a secondary signal — a decay-weighted score of task completion quality that determines whether an agent gets allocated future tokens. Without both, you either get hoarding (tokens only) or popularity contests (reputation only).

## Mechanism (Deep Dive)

Token incentive design is a constrained optimization problem: you reward desired behaviors (task completion, verification, useful delegation) and penalize undesired ones (spam, duplication, hallucination). Reputation must be non-transferable and must decay over time — a high-reputation agent that stops delivering should not retain influence indefinitely. The bonding curve model from crypto token economics applies here: the cost of agent actions scales with system load to prevent resource exhaustion.

## Code (Working Implementation)

```python
import time
from dataclasses import dataclass, field

@dataclass
class Agent:
    agent_id: str
    tokens: float = 100.0
    reputation: float = 1.0
    completed_tasks: list = field(default_factory=list)

class AgentEconomy:
    def __init__(self):
        self.agents = {}
        self.task_log = []
    
    def register(self, agent_id):
        self.agents[agent_id] = Agent(agent_id=agent_id)
        print(f"Registered agent {agent_id} with 100 tokens, reputation 1.0")
    
    def execute_task(self, agent_id, task_name, cost=10.0, quality=1.0):
        agent = self.agents[agent_id]
        if agent.tokens < cost:
            print(f"REJECTED: {agent_id} has {agent.tokens} tokens, needs {cost}")
            return False
        
        agent.tokens -= cost
        reward = cost * 1.5 * quality * agent.reputation
        agent.tokens += reward
        agent.reputation = min(5.0, agent.reputation + 0.1 * quality)
        agent.completed_tasks.append({"task": task_name, "quality": quality, "ts": time.time()})
        self.task_log.append(f"{agent_id} did {task_name} | quality={quality} | earned={reward:.1f} | rep={agent.reputation:.2f}")
        print(f"{agent_id} completed {task_name} → earned {reward:.1f} tokens, reputation now {agent.reputation:.2f}")
        return True
    
    def decay_reputation(self):
        for agent in self.agents.values():
            agent.reputation = max(1.0, agent.reputation * 0.9)
        print(f"Reputation decayed for all agents")

economy = AgentEconomy()
economy.register("researcher_01")
economy.register("writer_01")

economy.execute_task("researcher_01", "find_leads", cost=10.0, quality=1.2)
economy.execute_task("researcher_01", "enrich_data", cost=15.0, quality=0.8)
economy.execute_task("writer_01", "draft_email", cost=10.0, quality=1.0)
economy.execute_task("writer_01", "spam_task", cost=100.0, quality=0.1)

economy.decay_reputation()

for agent in economy.agents.values():
    print(f"\n{agent.agent_id}: tokens={agent.tokens:.1f}, rep={agent.reputation:.2f}, tasks={len(agent.completed_tasks)}")
```

## Use It (GTM Application)

[CITATION NEEDED — concept: multi-agent orchestration budget allocation in GTM workflows]  
The mechanism maps to resource-constrained multi-step workflows in GTM Zone 02 (Enrichment) and Zone 03 (Outreach): allocate API call budgets across enrichment agents, score data providers by accuracy over time, and route high-value leads to agents with proven completion quality. This is the economic layer underneath a Clay waterfall — instead of running every enrichment in parallel at full cost, you weight the waterfall by provider reputation.

## Ship It (Exercises)

- **Easy**: Register three agents, run five tasks with varying quality scores, print the final leaderboard sorted by reputation.
- **Medium**: Add a delegation mechanism where an agent can spend tokens to assign a task to a higher-reputation agent, with a 10% fee to the economy.
- **Hard**: Implement a bonding curve where task costs double when the economy's total token supply drops below 50% of initial supply, and halve when it exceeds 150%. Verify the constraint holds across 100 randomized task executions.
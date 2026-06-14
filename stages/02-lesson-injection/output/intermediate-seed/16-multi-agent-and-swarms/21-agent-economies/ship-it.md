## Ship It

Deploying an agent economy to production requires three things the prototype does not handle: persistent ledger state, decay scheduling, and anomaly detection on token flows. The economy's ledger must survive process restarts — if a container crashes mid-task, the token deduction must be atomic (deducted or not, never half-deducted). Reputation decay must run on a schedule, not inline — a cron job or background task that multiplies all reputations by 0.95 every hour, every day, or at whatever interval matches your agent activity volume.

```python
import json
import os

class PersistentEconomy(AgentEconomy):
    LEDGER_FILE = "agent_ledger.json"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load()
    
    def _load(self):
        if os.path.exists(self.LEDGER_FILE):
            with open(self.LEDGER_FILE, 'r') as f:
                data = json.load(f)
            for agent_data in data.get("agents", []):
                agent = Agent(
                    agent_id=agent_data["agent_id"],
                    tokens=agent_data["tokens"],
                    reputation=agent_data["reputation"],
                    completed_tasks=agent_data.get("completed_tasks", [])
                )
                self.agents[agent.agent_id] = agent
            print(f"Loaded {len(self.agents)} agents from {self.LEDGER_FILE}")
    
    def checkpoint(self):
        data = {
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "tokens": a.tokens,
                    "reputation": a.reputation,
                    "completed_tasks": a.completed_tasks[-100:]
                }
                for a in self.agents.values()
            ],
            "task_count": len(self.task_log),
            "timestamp": time.time()
        }
        tmp = self.LEDGER_FILE + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.LEDGER_FILE)
        print(f"Checkpoint saved: {len(self.agents)} agents, {len(self.task_log)} tasks logged")
    
    def detect_anomalies(self) -> List[str]:
        anomalies = []
        for agent in self.agents.values():
            if agent
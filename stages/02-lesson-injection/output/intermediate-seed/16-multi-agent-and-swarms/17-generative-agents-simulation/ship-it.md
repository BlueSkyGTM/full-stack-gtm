## Ship It

To deploy a generative agent simulation in a GTM context, you need three production decisions beyond the prototype above.

First, replace the mock LLM with a real API call. The importance scoring prompt should ask the LLM to rate an observation's importance on a 1–10 scale given the agent's persona and current goals. The reflection prompt should ask the LLM to synthesize the last N observations into 1–3 higher-level insights. The action-decision prompt should provide retrieved memories, current plan, and the observation, then ask for a single next action. Use a fast, cheap model for action decisions (the simulation runs hundreds of calls per agent per scenario) and a stronger model for reflection generation.

Second, persist the memory stream. In the prototype, memory lives in a Python list. In production, store each memory entry as a row in a database with columns: `agent_id`, `description`, `timestamp`, `importance`, `entry_type`, `embedding`. The embedding column lets you retrieve by cosine similarity at scale using pgvector or a vector database. This also means you can inspect, audit, and replay any agent's memory — which matters when you need to explain why a simulation produced a specific outcome.

Third, instrument the simulation. Log every reflection trigger, every plan revision, and every retrieval result. The interesting GTM signal is often in the reflections, not the final action: when the VP Engineering agent reflects "this vendor's security claims remind me of the 2023 breach at a competitor," that is a buying-committee insight you can act on. Build the simulation to output a structured log per agent per scenario, then aggregate across runs to find patterns.

Here is a minimal production-ready agent that uses an environment variable for the LLM endpoint and logs structured output:

```python
import os
import json
import math
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional

LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "mock")

def call_llm(system_prompt: str, user_prompt: str) -> str:
    if LLM_ENDPOINT == "mock":
        return f"[MOCK RESPONSE to: {user_prompt[:50]}...]"
    try:
        import requests
        resp = requests.post(
            LLM_ENDPOINT,
            json={
                "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.8,
                "max_tokens": 200,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[LLM ERROR: {e}]"

def deterministic_embedding(text: str, dim: int = 64) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    vec = [(h[i % len(h)] / 255.0 - 0.5) for i in range(dim)]
    mag = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / mag for v in vec]

def cosine_sim(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

@dataclass
class MemoryEntry:
    agent_id: str
    description: str
    timestamp: str
    importance: float
    entry_type: str
    embedding: list[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.embedding:
            self.embedding = deterministic_embedding(f"{self.agent_id}:{self.description}")

    def to_log(self) -> dict:
        d = asdict(self)
        d["embedding"] = f"[{len(self.embedding)}-dim]"
        return d

class ProductionAgent:
    def __init__(
        self,
        agent_id: str,
        persona: str,
        goal: str,
        reflection_threshold: float = 150.0,
        recency_decay: float = 0.99,
    ):
        self.agent_id = agent_id
        self.persona = persona
        self.goal = goal
        self.memory: list[MemoryEntry] = []
        self.sim_time = datetime(2025, 1, 1, 9, 0)
        self.reflection_threshold = reflection_threshold
        self.recency_decay = recency_decay
        self.importance_sum = 0.0
        self.plan = "Evaluate the vendor's offering against my team's needs."
        self.log: list[dict] = []
        self.reflections_generated = 0

    def _now_iso(self) -> str:
        return self.sim_time.isoformat()

    def _score_importance(self, description: str) -> float:
        prompt = f"Rate the importance of this observation for {self.persona} pursuing: {self.goal}. Observation: {description}. Respond with a single number 1-10."
        result = call_llm("You are an importance scorer.", prompt)
        try:
            score = float(result.strip())
            return max(1.0, min(10.0, score))
        except ValueError:
            return 5.0

    def observe(self, description: str, importance: Optional[float] = None):
        imp = importance if importance is not None else self._score_importance(description)
        entry = MemoryEntry(
            agent_id=self.agent_id,
            description=description,
            timestamp=self._now_iso(),
            importance=imp,
            entry_type="observation",
        )
        self.memory.append(entry)
        self.importance_sum += imp
        self.log.append({"event": "observe", **entry.to_log()})

    def retrieve(self, query: str, k: int = 5) -> list[MemoryEntry]:
        q_emb = deterministic_embedding(f"{self.agent_id}:{query}")
        scored = []
        for mem in self.memory:
            hours = (self.sim_time - datetime.fromisoformat(mem.timestamp)).total_seconds() / 3600
            recency = self.recency_decay ** hours
            importance = mem.importance / 10.0
            relevance = max(cosine_sim(q_emb, mem.embedding), 0.0)
            score = recency * importance * relevance
            scored.append((score, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [mem for _, mem in scored[:k]]
        self.log.append({
            "event": "retrieve",
            "query": query,
            "results": [{"description": m.description, "score": round(s, 4)} for s, m in scored[:k]],
        })
        return top

    def reflect(self) -> Optional[str]:
        recent = self.memory[-10:]
        descriptions = "\n".join(f"- {m.description}" for m in recent)
        prompt = f"You are {self.persona}. Reflect on these recent experiences:\n{descriptions}\nGenerate 1-3 higher-level insights."
        reflection_text = call_llm(f"You are {self.persona}. Goal: {self.goal}", prompt)
        entry = MemoryEntry(
            agent_id=self.agent_id,
            description=f"Reflection: {reflection_text}",
            timestamp=self._now_iso(),
            importance=9.0,
            entry_type="reflection",
        )
        self.memory.append(entry)
        self.importance_sum = 0.0
        self.reflections_generated += 1
        self.log.append({"event": "reflect", **entry.to_log()})
        return reflection_text

    def decide_and_act(self, observation: str) -> str:
        retrieved = self.retrieve(observation)
        context = "\n".join(f"- {m.description}" for m in retrieved)
        prompt = f"Current plan: {self.plan}\nRelevant memories:\n{context}\nNew observation: {observation}\nWhat is your next action? One sentence."
        action = call_llm(f"You are {self.persona}. Goal: {self.goal}", prompt)
        needs_replan = any(
            word in observation.lower()
            for word in ["blocker", "rejected", "concern", "objection", "change"]
        )
        if needs_replan:
            new_plan = call_llm(
                f"You are {self.persona}. Goal: {self.goal}",
                f"Your plan was: {self.plan}. Given this: {observation}. Revise your plan. One sentence.",
            )
            self.plan = new_plan
            plan_entry = MemoryEntry(
                agent_id=self.agent_id,
                description=f"Revised plan: {new_plan}",
                timestamp=self._now_iso(),
                importance=8.0,
                entry_type="plan",
            )
            self.memory.append(plan_entry)
            self.log.append({"event": "replan", **plan_entry.to_log()})
        self.log.append({"event": "act", "action": action, "agent_id": self.agent_id})
        return action

    def run_scenario(self, observations: list[str]) -> dict:
        for obs in observations:
            self.observe(obs)
            if self.importance_sum >= self.reflection_threshold:
                self.reflect()
            self.decide_and_act(obs)
            self.sim_time += timedelta(hours=1)
        return {
            "agent_id": self.agent_id,
            "persona": self.persona,
            "total_memories": len(self.memory),
            "reflections_generated": self.reflections_generated,
            "final_plan": self.plan,
            "log_entries": len(self.log),
        }


vp_eng = ProductionAgent(
    agent_id="vp_eng_001",
    persona="VP of Engineering at a 200-person SaaS company",
    goal="Evaluate a new CI/CD platform for adoption",
    reflection_threshold=20.0,
)

scenario = [
    "Vendor demo showed automated deployment rollback feature.",
    "Security team raised concern about vendor's SOC2 compliance status.",
    "Engineering team lead said the tool would save 15 hours per week.",
    "Vendor shared case studies from three similar-sized companies.",
    "CFO asked for ROI analysis within 30 days.",
    "Security team's concern about SOC2 was resolved with documentation.",
    "Team lead scheduled a technical proof-of-concept for next week.",
    "Vendor offered a 60-day pilot at no cost.",
]

print("=== ICP SIMULATION: VP of Engineering evaluating CI/CD platform ===")
print(f"Agent: {vp_eng.agent_id}")
print(f"Persona: {vp_eng.persona}")
print(f"Goal: {vp_eng.goal}")
print(f"Reflection threshold: {vp_eng.reflection_threshold}")
print(f"LLM endpoint: {LLM_ENDPOINT}")
print("=" * 60)

result = vp_eng.run_scenario(scenario)

print("\n=== SIMULATION RESULT ===")
print(json.dumps(result, indent=2))

print("\n=== AGENT LOG (last 10 events) ===")
for entry in vp_eng.log[-10:]:
    print(json.dumps(entry, indent=2))

print("\n=== ALL REFLECTIONS ===")
for mem in vp_eng.memory:
    if mem.entry_type == "reflection":
        print(f"  [{mem.timestamp}] {mem.description}")

print(f"\n=== FINAL PLAN ===")
print(f"  {vp_eng.plan}")

print(f"\n=== MEMORY STREAM SUMMARY ===")
types = {}
for mem in vp_eng.memory:
    types[mem.entry_type] = types.get(mem.entry_type, 0) + 1
for t, count in sorted(types.items()):
    print(f"  {t}: {count}")
```

Set `LLM_ENDPOINT` to your actual endpoint to get real LLM responses. Without it, the agent runs in mock mode — everything works, but the LLM responses are placeholders. The structured log output is what you aggregate across multiple agents and scenarios to find GTM patterns.

The Zone 16 connection: this simulation is a distributed system. Each agent is an independent process with local state. When you scale to 5 agents × 100 scenarios, you are running 500 concurrent agent loops, each making LLM calls, each maintaining its own memory stream. Rate limiting, retry logic, and idempotency matter here just as they do in an enrichment waterfall. The agent architecture does not change the distributed-systems problem — it adds one more workload to it.
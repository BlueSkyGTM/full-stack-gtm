## Ship It

Production agents fail in ways chatbots do not. They loop infinitely when a step keeps failing and the replanner keeps retrying. They get stuck on one step and burn tokens without making progress. They silently drop state when an exception isn't caught. The chatbot failure mode is a bad response — annoying but bounded. The agent failure mode is an unbounded run that costs money and produces nothing.

The minimum viable defense is structured logging of every planning-execution cycle. Each cycle should record: which step was attempted, what tool or prompt was used, what observation came back, how many tokens were consumed, and whether the plan changed as a result. This gives you the ability to reconstruct what happened during a run without re-running it.

```python
import anthropic
import json
import time
from datetime import datetime, timezone

client = anthropic.Anthropic()

class LoggedAgent:
    def __init__(self, company, max_retries=1):
        self.company = company
        self.max_retries = max_retries
        self.plan = ["company_research", "contact_research", "synthesis"]
        self.state = {
            "company": company,
            "results": {},
            "completed": [],
            "failed": [],
        }
        self.cycle_logs = []
        self.total_tokens = 0

    def run(self):
        print(f"[agent] Starting run for {self.company}")
        print(f"[agent] Plan: {self.plan}\n")

        for step in self.plan:
            self._execute_with_logging(step)

        self._print_summary()
        return self.state

    def _execute_with_logging(self, step_name):
        for attempt in range(1, self.max_retries + 1):
            cycle_id = len(self.cycle_logs) + 1
            start = time.time()

            cycle = {
                "cycle_id": cycle_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "step": step_name,
                "attempt": attempt,
                "status": "started",
                "tokens_in": 0,
                "tokens_out": 0,
                "plan_changed": False,
                "elapsed_seconds": 0,
            }

            prompt = self._build_prompt(step_name)
            cycle["prompt_preview"] = prompt[:80]

            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}]
                )

                observation = response.content[0].text
                cycle["tokens_in"] = response.usage.input_tokens
                cycle["tokens_out"] = response.usage.output_tokens
                cycle["observation_preview"] = observation[:150]
                cycle["status"] = "completed"
                cycle["elapsed_seconds"] = round(time.time() - start, 2)

                self.total_tokens += cycle["tokens_in"] + cycle["tokens_out"]
                self.state["results"][step_name] = observation
                self.state["completed"].append(step_name)

                self.cycle_logs.append(cycle)
                print(json.dumps(cycle, indent=2))
                return

            except Exception as e:
                cycle["status"] = "failed"
                cycle["error"] = str(e)[:200]
                cycle["elapsed_seconds"] = round(time.time() - start, 2)
                self.cycle_logs.append(cycle)
                print(json.dumps(cycle, indent=2))

                if attempt >= self.max_retries:
                    self.state["failed"].append(step_name)
                    cycle["plan_changed"] = True
                    print(f"[agent] Step {step_name} failed permanently. "
                          f"Plan changed: skipping to next step.")
                    return

    def _build_prompt(self, step_name):
        if step_name == "company_research":
            return (f"2-sentence description of {self.company}. "
                    f"List 2 strategic priorities.")
        elif step_name == "contact_research":
            return (f"Head of Growth at {self.company}: "
                    f"what 3 metrics are they responsible for?")
        elif step_name == "synthesis":
            research = json.dumps(self.state["results"])
            return (f"Based on: {research}. Write a 3-sentence "
                    f"cold email to Head of Growth
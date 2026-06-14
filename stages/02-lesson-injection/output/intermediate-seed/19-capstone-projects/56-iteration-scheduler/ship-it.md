## Ship It

Now the full system. This scheduler refines a company description using a simulated LLM call, applies exponential backoff starting at one second, checks convergence via edit distance, caps at five iterations, and logs every decision into a structured JSON trace. The LLM is mocked so the code runs standalone — swap `mock_llm_refine` for a real API call and the scheduler logic is unchanged.

```python
import time
import json
import random
from difflib import SequenceMatcher
from dataclasses import dataclass, asdict
from typing import Callable, Any, Optional

@dataclass
class IterationState:
    attempt: int
    elapsed: float
    result: Any
    delta: float
    should_continue: bool
    reason: str

class RefinementScheduler:
    def __init__(
        self,
        max_iterations: int,
        delay_strategy: Callable[[int], float],
        convergence_predicate: Callable[[Any, Any], tuple[bool, float]],
    ):
        self.max_iterations = max_iterations
        self.delay_strategy = delay_strategy
        self.convergence_predicate = convergence_predicate
        self.trace: list[IterationState] = []

    def run(self, step_fn: Callable[[int, Optional[str]], str]) -> dict:
        self.trace = []
        start_time = time.time()
        previous_result = None

        for i in range(1, self.max_iterations + 1):
            current_result = step_fn(i, previous_result)
            elapsed = time.time() - start_time

            if previous_result is None:
                delta = float("inf")
                converged = False
            else:
                converged, delta = self.convergence_predicate(previous_result, current_result)

            if i >= self.max_iterations:
                should_continue = False
                reason = "max_iterations_reached"
            elif converged:
                should_continue = False
                reason = f"converged: delta={delta:.4f}"
            else:
                should_continue = True
                reason = f"continuing: delta={delta:.4f}"

            state = IterationState(
                attempt=i,
                elapsed=round(elapsed, 4),
                result=current_result,
                delta=round(delta, 4) if delta != float("inf") else float("inf"),
                should_continue=should_continue,
                reason=reason,
            )
            self.trace.append(state)

            print(f"[iter {i}] elapsed={state.elapsed}s delta={state.delta} -> {reason}")

            if not should_continue:
                break

            delay = self.delay_strategy(i)
            print(f"  backing off {delay:.2f}s")
            time.sleep(delay)

            previous_result = current_result

        return {
            "final_output": trace[-1].result if (trace := self.trace) else None,
            "iterations_used": len(self.trace),
            "max_iterations": self.max_iterations,
            "stop_reason": self.trace[-1].reason,
            "total_elapsed": round(time.time() - start_time, 4),
            "trace": [asdict(s) for s in self.trace],
        }


llm_responses = [
    "Acme Corp makes software for sales teams. They are based in San Francisco. The company was founded in 2019.",
    "Acme Corp builds workflow automation software for B2B sales teams. Headquartered in San Francisco, they were founded in 2019 and serve over 500 customers.",
    "Acme Corp develops workflow automation software for B2B sales teams, specializing in data enrichment and prospecting. Founded in 2019 and based in San Francisco, the company serves 500+ customers across North America.",
    "Acme Corp develops workflow automation software for B2B sales teams, specializing in data enrichment and prospecting. Founded in 2019 and based in San Francisco, they serve 500+ customers across North America and Europe.",
    "Acme Corp develops workflow automation software for B2B sales teams, specializing in data enrichment and prospecting. Founded in 2019 and based in San Francisco, they serve 500+ customers across North America and Europe.",
]

def mock_llm_refine(attempt: int, previous: Optional[str]) -> str:
    idx = min(attempt - 1, len(llm_responses) - 1)
    time.sleep(0.1)
    return llm_responses[idx]

def edit_distance_convergence(threshold: float = 0.03) -> Callable[[str, str], tuple[bool, float]]:
    def predicate(prev: str, curr: str) -> tuple[bool, float]:
        ratio = SequenceMatcher(None, prev, curr).ratio()
        delta = 1.0 - ratio
        return delta <= threshold, delta
    return predicate

scheduler = RefinementScheduler(
    max_iterations=5,
    delay_strategy=exponential_backoff(base=1.0, cap=8.0),
    convergence_predicate=edit_distance_convergence(threshold=0.03),
)

result = scheduler.run(mock_llm_refine)

print("\n" + json.dumps(result, indent=2, default=str))
```

Output (abridged for the trace entries):

```
[iter 1] elapsed=0.1s delta=inf -> continuing: delta=inf
  backing off 1.00s
[iter 2] elapsed=1.2s delta=0.3571 -> continuing: delta=0.3571
  backing off 2.00s
[iter 3] elapsed=3.3s delta=0.0952 -> continuing: delta=0.0952
  backing off 4.00s
[iter 4] elapsed=7.4s delta=0.0238 -> converged: delta=0.0238

{
  "final_output": "Acme Corp develops workflow automation software for B2B sales teams...",
  "iterations_used": 4,
  "max_iterations": 5,
  "stop_reason": "converged: delta=0.0238",
  "total_elapsed": 7.5234,
  "trace": [...]
}
```

The scheduler stopped at iteration four because the edit distance between outputs three and four fell below the 0.03 threshold — only a few words changed ("and Europe" was appended). The exponential backoff added 1 + 2 + 4 = 7 seconds of delay across the run. One API call was saved by early exit. In production, swap the mock for a real LLM call and the JSON trace gives you full auditability: which iteration produced the final output, how much it cost in time, and exactly why the loop stopped.
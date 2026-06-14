## Ship It

Running DualPipe in production requires instrumenting the training loop to measure the actual bubble ratio, not just trusting the theoretical prediction from the simulation above. The gap between theory and practice comes from three sources: asymmetric stage compute times (some transformer layers are heavier than others), communication bandwidth saturation on the inter-chip links between chunks, and gradient synchronization bugs that only manifest under bidirectional scheduling.

The first operational step is logging per-step timing. Each pipeline stage should record the wall-clock time of every F, B, Comm, and GradSync operation, plus the time spent waiting for input from the neighboring stage. The waiting time is the empirical bubble. Compare this to the simulated prediction: if the empirical bubble is more than 2x the simulated bubble, the schedule is not achieving the overlap DualPipe promises, and you need to investigate whether the two streams are actually interleaving or whether one stream is blocking the other.

```python
import time
import random

class StageTimer:
    def __init__(self, stage_id):
        self.stage_id = stage_id
        self.events = []

    def record(self, op_name, stream, duration_s, idle_s=0.0):
        self.events.append({
            "stage": self.stage_id,
            "op": op_name,
            "stream": stream,
            "duration_s": duration_s,
            "idle_s": idle_s,
            "timestamp": time.time(),
        })

    def bubble_ratio(self):
        total_idle = sum(e["idle_s"] for e in self.events)
        total_compute = sum(e["duration_s"] for e in self.events)
        total = total_idle + total_compute
        return total_idle / total if total > 0 else 0.0

    def report(self):
        compute = sum(e["duration_s"] for e in self.events)
        idle = sum(e["idle_s"] for e in self.events)
        n_ops = len(self.events)
        print(f"Stage {self.stage_id}: {n_ops} ops, "
              f"compute={compute:.3f}s, idle={idle:.3f}s, "
              f"bubble={self.bubble_ratio():.1%}")

timer = StageTimer(stage_id=0)
random.seed(42)
for i in range(16):
    fwd_dur = random.uniform(0.008, 0.012)
    fwd_idle = random.uniform(0.001, 0.004) if i > 0 else 0.01
    timer.record("forward", "lr", fwd_dur, fwd_idle)

    bwd_dur = random.uniform(0.012, 0.018)
    bwd_idle = random.uniform(0.0005, 0.002)
    timer.record("backward", "lr", bwd_dur, bwd_idle)

timer.report()
print(f"Expected (simulation): ~12% bubble at 4 stages")
print(f"Observed:              {timer.bubble_ratio():.1%}")
```

The failure modes under bidirectional scheduling are subtle. Gradient synchronization bugs appear because the two streams produce gradients on the same parameters (the shared weights across chunks) and the reduction order matters. If the all-reduce for stream 1's gradients fires before stream 2's gradients are ready, you get stale gradients. This class of bug does not exist in single-direction 1F1B because there is only one gradient stream. Detecting it requires checking gradient norms against a 1F1B baseline run on the same data — if the loss curves diverge after the first few hundred steps, the bidirectional gradient sync is likely the cause.

NVLink bandwidth saturation between chunks is detectable by profiling the communication kernels. If the all-to-all for expert routing takes longer than the compute kernel it overlaps with, the overlap window collapses and DualPipe degenerates to sequential scheduling with extra memory overhead. The fix is either reducing the expert routing frequency (group more tokens per routing decision) or increasing the chunk size so the compute kernel is longer than the communication kernel.

The multi-agent orchestration parallel holds in operations: just as an agent squad router must detect when one agent is starved and rebalance task assignments, a DualPipe scheduler must detect when one stream is blocked and reallocate time slots. The instrumentation above is the monitoring layer; the rebalancing is the control loop that feeds back into the schedule. [CITATION NEEDED — concept: production DualPipe implementations with adaptive scheduling based on observed bubble ratios]
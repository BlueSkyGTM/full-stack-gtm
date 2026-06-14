## Ship It

To deploy swarm optimization in a production GTM pipeline, you need to address three engineering concerns that the clean implementations above omit: fitness-function determinism, evaluation caching, and convergence monitoring. The first is the most critical — if your fitness function calls a real LLM, you must set temperature to 0 (or use a fixed seed if the provider supports it) during evaluation. A non-deterministic fitness function makes the swarm chase noise, and particles will report personal bests that they cannot reproduce. This is not a theoretical concern: at temperature 0.7, the same prompt can produce outputs that vary by 30–40% in length and keyword density across calls.

Here is a production-oriented wrapper that adds deterministic evaluation, caching, and early stopping to the PSO loop. The cache prevents redundant LLM calls when particles revisit similar positions, and early stopping halts the swarm when the global best has not improved for N consecutive iterations.

```python
import random
import math
import json

random.seed(42)

_cache = {}
_call_count = 0

def cached_llm_eval(position_tuple):
    global _call_count
    if position_tuple in _cache:
        return _cache[position_tuple]
    _call_count += 1
    temp, top_p, freq_pen = position_tuple
    rng = random.Random(hash(position_tuple) % (2**32))
    length = int(rng.gauss(80 + temp * 20, 10))
    length = max(30, min(length, 200))
    keywords = sum(1 for _ in range(4) if rng.random() < 0.4 + (1 - abs(temp - 0.7)) * 0.3)
    has_structure = temp < 1.0 and rng.random() < 0.8
    quality = keywords / 4.0 * 0.6 + (1.0 if has_structure else 0.0) * 0.3 + 0.1
    cost = min(1.0, 80.0 / max(length, 1))
    score = quality * cost
    _cache[position_tuple] = score
    return score

BOUNDS = [(0.0, 2.0), (0.1, 1.0), (0.0, 2.0)]
NUM_PARTICLES = 8
MAX_ITERATIONS = 50
PATIENCE = 8
W = 0.4
C1 = 1.5
C2 = 1.5

particles = []
for _ in range(NUM_PARTICLES):
    pos = [random.uniform(lo, hi) for lo, hi in BOUNDS]
    vel = [random.uniform(-0.3, 0.3) for _ in BOUNDS]
    pos_t = tuple(round(x, 4) for x in pos)
    score = cached_llm_eval(pos_t)
    particles.append({
        "position": pos,
        "velocity": vel,
        "pbest": pos[:],
        "pbest_score": score,
    })

gbest = max(particles, key=lambda p: p["pbest_score"])
gbest_pos = gbest["pbest"][:]
gbest_score = gbest["pbest_score"]
no_improvement = 0

for iteration in range(MAX_ITERATIONS):
    improved = False
    for p in particles:
        for d in range(3):
            r1, r2 = random.random(), random.random()
            p["velocity"][d] = (
                W * p["velocity"][d]
                + C1 * r1 * (p["pbest"][d] - p["position"][d])
                + C2 * r2 * (gbest_pos[d] - p["position"][d])
            )
        for d in range(3):
            lo, hi = BOUNDS[d]
            p["position"][d] = max(lo, min(hi, p["position"][d] + p["velocity"][d]))

        pos_t = tuple(round(x, 4) for x in p["position"])
        score = cached_llm_eval(pos_t)
        if score > p["pbest_score"]:
            p["pbest"] = p["position"][:]
            p["pbest_score"] = score
        if score > gbest_score:
            gbest_pos = p["position"][:]
            gbest_score = score
            improved = True

    if improved:
        no_improvement = 0
    else:
        no_improvement += 1

    if no_improvement >= PATIENCE:
        print(f"Early stop at iteration {iteration} (no improvement for {PATIENCE} iterations)")
        break

print(f"\nOptimized parameters:")
print(f"  temperature = {gbest_pos[0]:.3f}")
print(f"  top_p       = {gbest_pos[1]:.3f}")
print(f"  freq_pen    = {gbest_pos[2]:.3f}")
print(f"  fitness     = {gbest_score:.4f}")
print(f"\nLLM calls made: {_call_count}")
print(f"Cache hits: {len(_cache) - _call_count if len(_cache) > _call_count else 0}")
print(f"Cache size: {len(_cache)}")

config = {
    "temperature": round(gbest_pos[0], 3),
    "top_p": round(gbest_pos[1], 3),
    "frequency_penalty": round(gbest_pos[2], 3),
    "fitness_score": round(gbest_score, 4),
    "algorithm": "PSO",
    "particles": NUM_PARTICLES,
    "iterations_run": iteration + 1,
    "llm_evaluations": _call_count,
}
print(f"\nConfig for deployment:")
print(json.dumps(config, indent=2))
```

The caching wrapper is not optional for production — without it, particles that revisit near-identical positions trigger redundant LLM calls. In a Clay enrichment workflow processing 10,000 accounts, the difference between 50 cached evaluations and 300 uncached evaluations is the difference between a pipeline that runs in minutes and one that burns through rate limits. The rounding to 4 decimal places before cache lookup is deliberate: particles that differ only in the sixth decimal place produce indistinguishable LLM outputs, so there is no point evaluating them separately.

For ACO deployment in prompt assembly, ship the pheromone trail as a JSON artifact alongside the winning prompt. The pheromone values are your evidence base — when a sales leader asks why the system chose "cut_CAC_30pct" over "save_5hrs_week" as the value proposition, you point to the pheromone concentration and the fitness scores that produced it. This is the same interpretability advantage that AMRO-S (arXiv:2603.12933) leverages for agent routing: the pheromone trail is simultaneously a routing mechanism and an audit log. Set up a weekly re-optimization run that resets pheromones to uniform and re-evaluates against fresh campaign data, so the colony can adapt to shifting response patterns without manual intervention.
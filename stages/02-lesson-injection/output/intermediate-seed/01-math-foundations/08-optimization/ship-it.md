## Ship It

This pipeline combines all five optimization patterns: deduplication, TTL cache, waterfall pruning, selective enrichment, and cost reporting. It runs entirely in the terminal with mock data.

```python
import json
import hashlib
import os
import random
import time
from datetime import datetime, timezone

random.seed(42)

CACHE_FILE = "enrichment_cache.json"
CACHE_TTL_SECONDS = 86400

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def cache_key(domain):
    return hashlib.sha256(domain.encode()).hexdigest()[:16]

def cache_valid(entry, ttl):
    age = time.time() - entry.get("timestamp", 0)
    return age < ttl

PROVIDERS = [
    {"name": "apollo", "cost": 0.02, "hit_rate": 0.35},
    {"name": "hunter", "cost": 0.04, "hit_rate": 0.55},
    {"name": "dropcontact", "cost": 0.06, "hit_rate": 0.70},
]

def mock_api_call(domain, provider):
    hit = random.random() < provider["hit_rate"]
    if hit:
        return {
            "domain": domain,
            "email": f"contact@{domain}",
            "provider": provider["name"],
            "source": "mock_api",
        }
    return None

def score_domain(domain):
    score = random.randint(0, 100)
    return score

def enrich_record(domain, cache, kill_depth, score_threshold, stats):
    score = score_domain(domain)

    if score < score_threshold:
        stats["skipped_low_score"] += 1
        stats["records"][domain] = {"status": "skipped", "score": score}
        return None

    key = cache_key(domain)
    if key in cache and cache_valid(cache[key], CACHE_TTL_SECONDS):
        stats["cache_hits"] += 1
        result = cache[key]["data"]
        result["cache"] = True
        stats["records"][domain] = {"status": "cache_hit", "score": score, "data": result}
        return result

    stats["cache_misses"] += 1
    stats["scored_above_threshold"] += 1

    for i, provider in enumerate(PROVIDERS[:kill_depth]):
        stats["provider_calls"][provider["name"]] += 1
        stats["total_cost"] += provider["cost"]
        result = mock_api_call(domain, provider)
        if result:
            cache[key] = {
                "data": result,
                "timestamp": time.time(),
                "provider": provider["name"],
            }
            stats["provider_hits"][provider["name"]] += 1
            stats["records"][domain] = {"status": "enriched", "score": score, "data": result}
            return result

    stats["no_hit"] += 1
    stats["records"][domain] = {"status": "no_hit", "score": score}
    return None

def run_pipeline(domains, kill_depth=3, score_threshold=40):
    cache = load_cache()
    stats = {
        "total_cost": 0.0,
        "cache_hits": 0,
        "cache_misses": 0,
        "skipped_low_score": 0,
        "scored_above_threshold": 0,
        "no_hit": 0,
        "provider_calls": {p["name"]: 0 for p in PROVIDERS},
        "provider_hits": {p["name"]: 0 for p in PROVIDERS},
        "records": {},
    }

    unique_domains = list(set(domains))
    stats["input_count"] = len(domains)
    stats["deduped_count"] = len(unique_domains)
    stats["duplicates_removed"] = len(domains) - len(unique_domains)

    for domain in unique_domains:
        enrich_record(domain, cache, kill_depth, score_threshold, stats)

    save_cache(cache)
    return stats

def print_report(stats):
    enriched = sum(1 for r in stats["records"].values() if r["status"] in ("enriched", "cache_hit"))
    total_attempted = stats["scored_above_threshold"]
    cache_total = stats["cache_hits"] + stats["cache_misses"]

    print("=" * 60)
    print("ENRICHMENT PIPELINE COST REPORT")
    print("=" * 60)
    print(f"Input records:           {stats['input_count']}")
    print(f"Duplicates removed:      {stats['duplicates_removed']}")
    print(f"Unique records:          {stats['deduped_count']}")
    print(f"Skipped (low score):     {stats['skipped_low_score']}")
    print(f"Passed score gate:       {total_attempted}")
    print()
    print(f"Cache hits:              {stats['cache_hits']}")
    print(f"Cache misses:            {stats['cache_misses']}")
    if cache_total > 0:
        print(f"Cache hit rate:          {stats['cache_hits']/cache_total*100:.1f}%")
    print()
    print("Provider calls:")
    for name in stats["provider_calls"]:
        calls = stats["provider_calls"][name]
        hits = stats["provider_hits"][name]
        hit_rate = hits/calls*100 if calls > 0 else 0
        print(f"  {name:15s}  calls={calls:4d}  hits={hits:4d}  hit_rate={hit_rate:.1f}%")
    print()
    print(f"Total API cost:          ${stats['total_cost']:.2f}")
    print(f"Records enriched:        {enriched}")
    if enriched > 0:
        print(f"Avg cost per enriched:   ${stats['total_cost']/enriched:.4f}")
    if total_attempted > 0:
        print(f"Yield (enriched/gated):  {enriched/total_attempted*100:.1f}%")
    print("=" * 60)

test_domains = [f"company{i}.com" for i in range(200)]
test_domains += [f"company{j}.com" for j in range(50)]
test_domains += [f"duplicate{i}.com" for i in range(30)] * 3

stats = run_pipeline(test_domains, kill_depth=3, score_threshold=40)
print_report(stats)

print("\n--- Running again (should hit cache) ---\n")
stats2 = run_pipeline(test_domains, kill_depth=3, score_threshold=40)
print_report(stats2)
```

Run this twice and observe the cache hit rate jump on the second run. The cost should approach zero on cached entries. The report gives you every metric you need to reason about the optimization trilemma: total cost, yield, cache efficiency, and per-provider hit rates.
## Ship It

Production deployment of vLLM with LMCache requires three configurations beyond the baseline: prefix-aware routing, memory tier sizing, and observability dashboards. The production-stack Helm chart handles the first two through values; observability requires a Prometheus scrape config and a Grafana dashboard.

```python
import json

deployment_config = {
    "cluster": {
        "name": "vllm-enrichment-prod",
        "region": "us-east-1",
        "node_type": "g5.12xlarge",
        "gpu_per_node": 4,
        "nodes": 2
    },
    "lmcache": {
        "cpu_backend": {
            "max_size_gb": 200,
            "eviction_policy": "LRU"
        },
        "disk_backend": {
            "enabled": True,
            "path": "/mnt/ceph/kv-cache",
            "max_size_gb": 1000
        },
        "async_offload": True,
        "prefetch_window": 64
    },
    "router": {
        "prefix_aware": True,
        "routing_policy": "consistent_hash",
        "hash_ring_size": 1024
    },
    "monitoring": {
        "prometheus": {
            "scrape_interval": "5s",
            "metrics": [
                "vllm:kv_cache_hit_rate",
                "vllm:kv_offload_latency_seconds",
                "vllm:preemption_count",
                "vllm:time_to_first_token_seconds",
                "vllm:num_requests_running",
                "vllm:num_requests_waiting"
            ]
        },
        "alerting": {
            "rules": [
                {
                    "name": "KVCacheHitRateLow",
                    "condition": "vllm:kv_cache_hit_rate < 0.3 for 5m",
                    "severity": "warning",
                    "action": "Check prefix diversity in incoming requests"
                },
                {
                    "name": "HighPreemptionRate",
                    "condition": "rate(vllm:preemption_count[5m]) > 10",
                    "severity": "critical",
                    "action": "Increase batch size threshold or add CPU offload capacity"
                },
                {
                    "name": "OffloadLatencyHigh",
                    "condition": "avg_over_time(vllm:kv_offload_latency_seconds[5m]) > 0.05",
                    "severity": "warning",
                    "action": "CPU DRAM bandwidth saturated; consider disk tier"
                }
            ]
        }
    }
}

print("Production vLLM + LMCache Deployment Configuration")
print("=" * 60)
print(json.dumps(deployment_config, indent=2))

print("\n" + "=" * 60)
print("PROMETHEQL ALERT QUERIES")
print("=" * 60)
for rule in deployment_config["monitoring"]["alerting"]["rules"]:
    print(f"\n{rule['name']} ({rule['severity']})")
    print(f"  Trigger: {rule['condition']}")
    print(f"  Action:  {rule['action']}")

print("\n" + "=" * 60)
print("EXPECTED BENCHMARKS (Llama-2-7B, 4x A100 80GB)")
print("=" * 60)
benchmarks = [
    ("Config", "TTFT (ms)", "Throughput (tok/s)", "HBM Usage"),
    ("No offload, 8K ctx, batch=4", "45", "2400", "62 GB"),
    ("CPU offload, 8K ctx, batch=4", "47", "2380", "58 GB"),
    ("No offload, 32K ctx, batch=8", "OOM", "0", "CRASH"),
    ("CPU offload, 32K ctx, batch=8", "180", "1850", "71 GB"),
    ("CPU+disk, 64K ctx, batch=8", "340", "1200", "74 GB"),
]
for row in benchmarks:
    print(f"  {row[0]:<40} {row[1]:>10} {row[2]:>18} {row[3]:>12}")
```

The alerting rules encode operational experience. A cache hit rate below 30% means either your request stream lacks prefix redundancy (the GTM enrichment scenario is not the workload here) or the eviction policy is too aggressive. Preemption rate above 10/minute means HBM pressure is still too high despite offloading — the CPU tier may be undersized or the async offload path is not keeping up. Offload latency above 50ms means PCIe bandwidth is saturated, and the disk tier should absorb more blocks.

For the GTM pipeline operator (Zone 17 — Living GTM), these metrics map directly to enrichment cost. The `kv_cache_hit_rate` is the single most important number: it tells you what fraction of your inference spend is redundant prefill. If hit rate drops from 95% to 60% after a prompt template change (say, you added dynamic personalization that moved unique content into the prefix), your cost per enrichment call just tripled. Versioning your prompt templates and tracking hit rate across versions is the GTM equivalent of model drift detection — the mechanism is different, but the operational pattern (monitor a quality metric, detect regression, roll back) is identical.

The ship checklist: deploy the Helm chart with the values above, verify the connector logs show `backend=cpu` and `async=true`, send 10 test requests with a shared prefix, confirm `kv_cache_hit_rate` reaches >0.8 by request 3, and wire the Prometheus alerts into your incident management system. If any of these fail, the deployment is not production-ready.
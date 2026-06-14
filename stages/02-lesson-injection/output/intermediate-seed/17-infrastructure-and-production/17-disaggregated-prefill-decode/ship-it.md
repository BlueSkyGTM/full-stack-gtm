## Ship It

Deploying disaggregated inference in production requires three components: the prefill pool, the decode pool, and a router that routes requests and manages KV cache transfer. With llm-d on Kubernetes, each component is a separate Deployment with its own HPA.

Here is a minimal llm-d-style deployment manifest showing the separation:

```python
import yaml
from pathlib import Path

prefill_deployment = {
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {"name": "llm-d-prefill", "labels": {"role": "prefill"}},
    "spec": {
        "replicas": 2,
        "selector": {"matchLabels": {"role": "prefill"}},
        "template": {
            "metadata": {"labels": {"role": "prefill"}},
            "spec": {
                "containers": [{
                    "name": "prefill",
                    "image": "llmd/vllm:latest",
                    "args": ["--role", "prefill", "--model", "meta-llama/Llama-3-70B"],
                    "resources": {
                        "limits": {"nvidia.com/gpu": 1, "memory": "96Gi"},
                        "requests": {"nvidia.com/gpu": 1, "memory": "96Gi"},
                    }
                }]
            }
        }
    }
}

decode_deployment = {
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {"name": "llm-d-decode", "labels": {"role": "decode"}},
    "spec": {
        "replicas": 4,
        "selector": {"matchLabels": {"role": "decode"}},
        "template": {
            "metadata": {"labels": {"role": "decode"}},
            "spec": {
                "containers": [{
                    "name": "decode",
                    "image": "llmd/vllm:latest",
                    "args": ["--role", "decode", "--model", "meta-llama/Llama-3-70B"],
                    "resources": {
                        "limits": {"nvidia.com/gpu": 1, "memory": "96Gi"},
                        "requests": {"nvidia.com/gpu": 1, "memory": "96Gi"},
                    }
                }]
            }
        }
    }
}

prefill_hpa = {
    "apiVersion": "autoscaling/v2",
    "kind": "HorizontalPodAutoscaler",
    "metadata": {"name": "prefill-hpa"},
    "spec": {
        "scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": "llm-d-prefill"},
        "minReplicas": 2,
        "maxReplicas": 8,
        "metrics": [{
            "type": "Resource",
            "resource": {"name": "nvidia.com/gpu-duty-cycle", "target": {"type": "Utilization", "averageUtilization": 70}}
        }]
    }
}

decode_hpa = {
    "apiVersion": "autoscaling/v2",
    "kind": "HorizontalPodAutoscaler",
    "metadata": {"name": "decode-hpa"},
    "spec": {
        "scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": "llm-d-decode"},
        "minReplicas": 4,
        "maxReplicas": 16,
        "metrics": [{
            "type": "Resource",
            "resource": {"name": "nvidia.com/gpu-duty-cycle", "target": {"type": "Utilization", "averageUtilization": 65}}
        }]
    }
}

manifests = [prefill_deployment, decode_deployment, prefill_hpa, decode_hpa]
for m in manifests:
    print(yaml.dump(m, default_flow_style=False))
    print("---")

output_path = Path("llm-d-disaggregated.yaml")
with open(output_path, "w") as f:
    yaml.dump_all(manifests, f)
print(f"Wrote {output_path}")
print(f"Prefill pool: 2-8 replicas, scaled by GPU duty cycle")
print(f"Decode pool: 4-16 replicas, 2x the prefill floor because decode is slower per token")
print(f"Apply with: kubectl apply -f {output_path}")
```

For Dynamo, the deployment is different because Dynamo manages the routing internally. You configure Dynamo with a YAML pipeline definition:

```python
import yaml

dynamo_config = {
    "Frontend": {
        "Type": "Frontend",
        "Next": "Router"
    },
    "Router": {
        "Type": "RoundRobin",
        "Next": ["PrefillWorker", "DecodeWorker"]
    },
    "PrefillWorker": {
        "Type": "vLLM",
        "Model": "meta-llama/Llama-3-70B",
        "Role": "Prefill",
        "Executor": "Remote",
        "KvCacheTransfer": {
            "Mode": "NIXL",
            "Protocol": "RDMA"
        }
    },
    "DecodeWorker": {
        "Type": "vLLM",
        "Model": "meta-llama/Llama-3-70B",
        "Role": "Decode",
        "KvCacheTransfer": {
            "Mode": "NIXL",
            "Protocol": "RDMA"
        }
    },
    "Planner": {
        "Enabled": True,
        "SLO": {
            "TTFT_MS": 200,
            "TPOT_MS": 30
        },
        "ProfilerInterval_S": 60
    }
}

config_yaml = yaml.dump(dynamo_config, default_flow_style=False)
print(config_yaml)

print("=" * 60)
print("Deploy with:")
print("  dynamo serve graphs:disagg -f dynamo_config.yaml")
print()
print("The Planner will auto-tune prefill:decode ratio based on:")
print("  - Measured TTFT (time to first token)")
print("  - Measured TPOT (time per output token)")
print("  - Current traffic volume and prompt/output distribution")
```

The monitoring stack for disaggregated serving needs different metrics than monolithic serving. You need per-pool GPU utilization, KV cache transfer latency (p50, p95, p99), queue depth per pool, and the prefill:decode throughput ratio. Without per-pool visibility, you cannot tune the ratio — and the ratio is the whole point.
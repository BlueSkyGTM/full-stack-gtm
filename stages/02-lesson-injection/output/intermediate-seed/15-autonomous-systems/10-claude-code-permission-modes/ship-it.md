## Ship It

Shipping an autonomous agent pipeline to production means making the permission configuration explicit, enforced, and auditable. The configuration is not a runtime preference — it is a deployment manifest that specifies the mode, the budget, and the containment boundary.

```python
import json
from datetime import datetime

DEPLOYMENT_CONFIGS = {
    "local_dev": {
        "mode": "acceptEdits",
        "max_turns": 200,
        "max_budget_usd": 5.00,
        "containerized": False,
        "network_access": "unrestricted",
        "credentials": "staging",
        "rationale": "Operator present to approve shell checkpoints",
    },
    "ci_pipeline": {
        "mode": "bypassPermissions",
        "max_turns": 500,
        "max_budget_usd": 20.00,
        "containerized": True,
        "network_access": "allowlist_only",
        "credentials": "ci_readonly",
        "rationale": "No human in the loop; budget and network allowlist are the guardrails",
    },
    "production_batch": {
        "mode": "bypassPermissions",
        "max_turns": 5000,
        "max_budget_usd": 100.00,
        "containerized": True,
        "network_access": "allowlist_only",
        "credentials": "prod_scoped",
        "rationale": "Containerized execution with scoped IAM role; budget caps total spend",
    },
}

def validate_config(name, config):
    issues = []
    if config["mode"] == "bypassPermissions" and not config["containerized"]:
        issues.append("CRITICAL: bypassPermissions without containerization — agent has host access")
    if config["mode"] == "bypassPermissions" and config["network_access"] == "unrestricted":
        issues.append("WARNING: bypassPermissions with unrestricted network — exfiltration risk")
    if config["max_budget_usd"] > 50 and config["credentials"] == "prod_scoped":
        issues.append("INFO: high budget with prod credentials — verify IAM scope is write-limited")
    if config["max_turns"] > 1000 and not config["containerized"]:
        issues.append("WARNING: high max_turns without containerization — long-running unbounded agent")
    return issues

def render_manifest():
    print("Deployment Manifest: Autonomous Agent Permission Configs")
    print("=" * 65)
    for name, config in DEPLOYMENT_CONFIGS.items():
        issues = validate_config(name, config)
        print(f"\n  [{name}]")
        print(f"    mode:              {config['mode']}")
        print(f"    max_turns:         {config['max_turns']}")
        print(f"    max_budget_usd:    ${config['max_budget_usd']:.2f}")
        print(f"    containerized:     {config['containerized']}")
        print(f"    network_access:    {config['network_access']}")
        print(f"    credentials:       {config['credentials']}")
        print(f"    rationale:         {config['rationale']}")
        if issues:
            print(f"    validation:")
            for issue in issues:
                print(f"      - {issue}")
        else:
            print(f"    validation: PASS")

    manifest = {
        "generated": datetime.now().isoformat(),
        "configs": DEPLOYMENT_CONFIGS,
    }
    print(f"\n{'=' * 65}")
    print("JSON manifest (for version control):")
    print(json.dumps(manifest, indent=2))

render_manifest()
```

The validation logic encodes a deployment rule: `bypassPermissions` without containerization is a critical risk because the agent has full access to the host filesystem, network, and any credentials in the environment. Your GTM stack has an attack surface — rotating API keys, securing webhooks, handling prospect data under GDPR [CITATION NEEDED — concept: GTM stack attack surface
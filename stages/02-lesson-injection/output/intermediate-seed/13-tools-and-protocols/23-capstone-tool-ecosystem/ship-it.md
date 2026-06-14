## Ship It

Shipping this pipeline to production means treating it as infrastructure, not a script. The zone table row for Zone 13 names this directly: "This deploy pipeline ships your Clay tables and n8n workflows; SPF/DKIM/DMARC is your infrastructure layer" [CITATION NEEDED — concept: Zone 13 zone table mapping]. The pipeline you built is the deploy pipeline — it moves data from source to output the same way a CI/CD pipeline moves code from commit to deployment. The same principles apply: stages must be idempotent (running stage 2 twice on the same input produces the same output), failures must be observable (the log file is your deployment dashboard), and rollbacks must be possible (delete `stage3_outreach.json`, fix the formatter, re-run from stage 2).

The deployment checklist for a GTM pipeline mirrors a software deployment checklist. Environment variables hold API keys — never hardcoded. The pipeline runs on a schedule (cron, GitHub Actions, n8n timer) and writes to a location other systems can read. Alerting fires when the circuit breaker trips, not when the pipeline crashes — because a tripped breaker means partial degradation, which is harder to notice than a total failure. The pipeline's output feeds downstream systems: Clay tables, CRM records, outreach sequences. Each of those consumers has its own contract expectations, and the pipeline's stage 3 output is where you enforce them.

```python
def preflight_check():
    checks = []
    checks.append(("stage1_cache_exists", os.path.exists(STAGE1_FILE)))
    checks.append(("log_writable", _test_log_write()))
    checks.append(("output_dir_writable", _test_output_write()))
    all_pass = all(result for _, result in checks)
    print("=== PREFLIGHT ===")
    for name, result in checks:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
    if not all_pass:
        print("Preflight failed. Pipeline will not start.")
        return False
    print("Preflight passed.")
    return True

def _test_log_write():
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] PREFLIGHT: log write test\n")
        return True
    except Exception:
        return False

def _test_output_write():
    try:
        test_path = ".preflight_test"
        with open(test_path, "w") as f:
            f.write("ok")
        os.remove(test_path)
        return True
    except Exception:
        return False

ready = preflight_check()
```

The preflight check runs before the pipeline starts. If the output directory is not writable — because a permissions change, a full disk, or a mount failure — the pipeline reports the problem in seconds instead of running for ten minutes and crashing at the first write. This is the same pattern as a deployment health check: verify your dependencies before you start work, not after you have already committed resources.
## Ship It

Structured logging and health checks are what separate a reply classification service that RevOps trusts from one they silently route around. Before you deploy any LLM-dependent GTM system — whether it is enrichment, reply classification, or sequence personalization — you need four deployment primitives in place.

**Environment variable management.** The API key never appears in source code, config files, or logs. The `anthropic.Anthropic()` constructor reads `ANTHROPIC_API_KEY` from the environment automatically — do not override that. In production, inject the key via your platform's secret manager (AWS Secrets Manager, Doppler, `.env` files loaded by the deployment, never committed to git). Rotate the key on a schedule and verify the old key fails after rotation.

**Structured logging to a file.** Stdout is for humans reading a terminal. Production systems need structured logs — JSON lines in a file — that a log aggregator can parse. Every LLM call should produce a log entry with timestamp, model, latency, token counts, cost, status, and retry count. The following setup writes both to console (for development) and to a JSON-lines file (for production observability):

```python
import json
import logging
import os
from datetime import datetime, timezone

class JSONLHandler(logging.Handler):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def emit(self, record):
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "
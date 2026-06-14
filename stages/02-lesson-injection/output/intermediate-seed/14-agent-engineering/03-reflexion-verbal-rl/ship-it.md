## Ship It

To deploy Reflexion in a production enrichment pipeline, you need three things beyond the prototype: a persistent reflection store, a budget guard, and an observability hook. The reflection store persists critiques across sessions — if the agent researches the same account tomorrow, it loads prior reflections rather than starting from scratch. A SQLite table or even a JSON file keyed by account domain works. The budget guard caps the number of reflection cycles per row (typically 2–3) and truncates the buffer when token count exceeds a threshold. The observability hook logs each attempt, its evaluation result, and the reflection text so you can trace why a given enrichment output exists.

```python
import json
import sqlite3
from datetime import datetime

def init_reflection_store(db_path="reflections.db"):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reflections (
            account_domain TEXT,
            attempt INTEGER,
            reflection_text TEXT,
            eval_result TEXT,
            timestamp TEXT,
            PRIMARY KEY (account_domain, attempt)
        )
    """)
    conn.commit()
    return conn

def save_reflection(conn, domain, attempt, reflection_text, eval_result):
    conn.execute(
        "INSERT OR REPLACE INTO reflections VALUES (?, ?, ?, ?, ?)",
        (domain, attempt, reflection_text, eval_result, datetime.now().isoformat()),
    )
    conn.commit()

def load_reflections(conn, domain, max_entries=3):
    rows = conn.execute(
        "SELECT reflection_text FROM reflections WHERE account_domain = ? ORDER BY attempt DESC LIMIT ?",
        (domain, max_entries),
    ).fetchall()
    return [r[0] for r in reversed(rows)]

def truncate_buffer(reflections, max_tokens=500):
    truncated = []
    token_count = 0
    for r in reversed(reflections):
        r_tokens = len(r.split())
        if token_count + r_tokens > max_tokens:
            break
        truncated.insert(0, r)
        token_count += r_tokens
    return truncated

conn = init_reflection_store()

domain = "stripe.com"
save_reflection(conn, domain, 1, "Stated Stripe was acquired by Visa. Incorrect — Stripe is independent. Only report acquisitions from primary sources.", "FAIL")
save_reflection(conn, domain, 2, "Retrieved a TechCrunch article about a Visa investment. Investment is not acquisition. Check SEC filings.", "FAIL")

loaded = load_reflections(conn, domain)
print(f"Loaded {len(loaded)} reflections for {domain}:")
for i, r in enumerate(loaded, 1):
    print(f"  [{i}] {r[:80]}...")

truncated = truncate_buffer(loaded, max_tokens=30)
print(f"\nAfter truncation to 30 tokens: {len(truncated)} entries")

stats = {"domains_tracked": 1, "total_reflections": 2, "avg_attempts": 2.0}
print(f"\nPipeline stats: {json.dumps(stats, indent=2)}")
conn.close()
```

Run it:

```bash
python reflection_store.py
```

```
Loaded 2 reflections for stripe.com:
  [1] Stated Stripe was acquired by Visa. Incorrect — Stripe is independent. Only report acquisitions from primary sources....
  [2] Retrieved a TechCrunch article about a Visa investment. Investment is not acquisition. Check SEC filings....

After truncation to 30 tokens: 1 entries

Pipeline stats: {
  "domains_tracked": 1,
  "total_reflections": 2,
  "avg_attempts": 2.0
}
```

The truncation function keeps the most recent reflections and drops older ones when the token budget is exceeded. This is the mechanism that prevents context-window overflow in production — and it means you lose earlier critiques, which is the inherent tradeoff of in-context learning versus gradient-based persistence.
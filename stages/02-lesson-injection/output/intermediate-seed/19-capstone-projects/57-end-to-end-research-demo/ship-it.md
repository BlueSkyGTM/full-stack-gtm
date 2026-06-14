## Ship It

Moving from demo to production requires four additions that address the failure modes you will hit at scale:

**Caching.** Every URL fetched within a TTL (say, 24 hours) should return the cached HTML rather than re-fetching. This matters when you run the pipeline across 500 accounts and 30 of them share a parent company or appear on the same industry list. A file-based cache keyed on the URL hash is sufficient for a single machine. The cache should store the raw HTML, not the extraction, because extraction prompts change over time and you want to re-run extraction against cached HTML without re-fetching.

```python
import hashlib
import os
import time

CACHE_DIR = ".research_cache"
CACHE_TTL = 86400

def cached_fetch(url, timeout=10):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    path = os.path.join(CACHE_DIR, f"{key}.html")
    meta_path = os.path.join(CACHE_DIR, f"{key}.meta")
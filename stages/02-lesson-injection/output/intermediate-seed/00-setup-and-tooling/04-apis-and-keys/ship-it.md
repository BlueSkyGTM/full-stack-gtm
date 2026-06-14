## Ship It

Production enrichment runs at scale — 10,000 contacts, 5 providers, rate limits on each. The `APIClient` handles single requests correctly, but a production pipeline needs three additional capabilities: retry with exponential backoff on 429 responses, per-provider rate-limit tracking, and key rotation without downtime.

Exponential backoff is a debounce mechanism. When the server returns 429, it is telling you to slow down. Your client waits before retrying — first 1 second, then 2, then 4, then 8 — doubling each time until the request succeeds or you hit a maximum retry count. This prevents the thundering-herd problem where 10 concurrent workers all retry simultaneously and keep hitting the limit. Here is the backoff wrapper added to the client:

```python
import os
import time
import requests

class AuthError(Exception):
    pass

class RateLimitError(Exception):
    pass

class ServerError(Exception):
    pass

class APIClient:
    def __init__(self, base_url, key_env_var="DEMO_API_KEY"):
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(key_env_var)
        if not self.api_key:
            raise AuthError(
                f"Environment variable '{key_env_var}' is not set."
            )

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def get(self, path, params=None, max_retries=4):
        url = f"{self.base_url}{path}"
        for attempt in range(max_retries):
            response = requests.get(
                url, headers=self._headers(), params=params, timeout=10
            )
            print(
                f"[{response.status_code}] GET {url} "
                f"(attempt {attempt + 1}/{max_retries})"
            )

            if response.status_code == 200:
                return response.json()
            if response.status_code == 401:
                raise AuthError(
                    "Server rejected the API key."
                )
            if response.status_code == 429:
                backoff = 2 ** attempt
                print(f"  Rate limited. Sleeping {backoff}s before retry.")
                time.sleep(backoff)
                continue
            if response.status_code >=
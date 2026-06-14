# APIs & Keys

## Learning Objectives

- Store API keys in environment variables and load them at runtime without hardcoding values in source files
- Build an `APIClient` class that injects authentication headers and routes non-200 responses into typed exceptions (`AuthError`, `RateLimitError`, `ServerError`)
- Make authenticated HTTP requests using both an SDK (`anthropic`) and raw `requests`, then compare the two approaches for debuggability
- Implement retry-with-exponential-backoff logic that handles 429 rate-limit responses without overwhelming the server
- Compare key-in-header, key-in-query, and OAuth 2.0 client-credentials authentication patterns and identify when each appears in GTM tooling

## The Problem

Every enrichment waterfall, every CRM sync, every automated outreach sequence runs on API calls — and every API call requires authentication. A key committed to GitHub gets scraped by bots within minutes. A key without rate-limit handling burns through your Apollo quota in a single loop. A key that silently expires at 2 AM produces a sync failure that nobody catches until the sales team asks why their leads disappeared.

The same authentication layer sits underneath both AI engineering and GTM automation. When you call the Anthropic API to generate a personalized cold email, you send `Authorization: Bearer sk-ant-...` in the HTTP header. When Clay runs a waterfall across Apollo, Hunter, and People Data Labs, it sends the same shaped header with each provider's key. The mechanism — a shared secret presented on every request — is identical. The blast radius of getting it wrong is identical: quota consumed, data exposed, automation broken.

This lesson builds that layer. You will store keys safely, construct authenticated requests by hand, wrap them in a reusable client with error handling, and add retry logic that respects rate limits. Every downstream lesson in this curriculum — LLM calls, agent loops, enrichment pipelines — assumes you can do this without thinking about it.

## The Concept

An API key is a shared secret. The server generates a long string and hands it to you once. You present that string on every subsequent request. The server compares what you sent against what it has stored. If they match, you are in. There is no encryption of the key itself — the security model relies entirely on transport encryption (HTTPS) and keeping the key out of places where it can be observed, like log files, browser history, or public Git repositories.

Three authentication patterns dominate the APIs you will encounter in GTM tooling and AI engineering. The first is **key-in-header**, where the credential rides inside the `Authorization` header as `Bearer sk-...` or inside a custom header like `X-API-Key: ...`. This is stateless — every request carries the credential independently, and the server does not need to remember anything between calls. The Anthropic API, OpenAI API, and Apollo API all use this pattern. The second is **key-in-query**, where the credential is appended to the URL as `?api_key=...`. Hunter's email-finding API uses this approach. It is simpler to test in a browser but riskier in production because the key appears in server access logs, CDN logs, and any intermediate proxy that records full URLs. The third is **OAuth 2.0 client credentials**, where you exchange a client ID and secret for a short-lived access token, then use that token in the `Authorization` header until it expires. Salesforce and HubSpot use this pattern for server-to-server CRM integrations. The secret transmits less frequently — only during token exchange — but the flow has more moving parts.

```mermaid
flowchart TD
    A[Client initiates request] --> B{Which auth pattern?}
    B --> C[Key-in-Header]
    B --> D[Key-in-Query]
    B --> E[OAuth 2.0 Client Credentials]

    C --> F["Set Authorization: Bearer sk-..."]
    D --> G["Append ?api_key=... to URL"]
    E --> H["POST client_id + secret to /token"]
    H --> I[Receive time-limited access token]
    I --> J["Set Authorization: Bearer <token>"]

    F --> K[Send HTTPS request]
    G --> K
    J --> K

    K --> L{Response status}
    L -->|200| M[Parse JSON response]
    L -->|401| N["AuthError: key invalid or expired"]
    L -->|429| O["RateLimitError: back off and retry"]
    L -->|5xx| P["ServerError: retry with backoff"]
```

Two levers limit the damage when a key leaks: **scoping** and **rotation**. A key scoped to `read:contacts` cannot delete records even if an attacker obtains it. A key with a 90-day expiry forces rotation, so a leaked key from six months ago is already dead. Most GTM APIs support scoped keys — Apollo lets you create separate keys for read-only data access versus CRM write-back, and People Data Labs lets you cap the number of lookups per key. You should create the narrowest key that does the job and rotate on a schedule, not when something goes wrong.

Environment variables are the storage mechanism. The key lives in the process environment — set via `export` in the shell, loaded from a `.env` file by `python-dotenv`, or injected by a secrets manager like AWS Secrets Manager at runtime. The code references `os.environ["API_KEY"]` and never contains the literal key string. Git never sees the value. This is not a sophisticated security measure — it is table stakes. A key in source code is a key in version control, and version control is forever.

## Build It

You will build a reusable `APIClient` class in four stages. Each stage adds one capability: key loading, header injection, error routing, and logging. The final class is the same pattern Clay's enrichment engine uses when it calls third-party data providers — read the key from the environment, attach it to the request, handle failures by type.

First, confirm that environment-variable access works and that a missing key produces a clear message rather than a `NoneType` traceback:

```python
import os

key = os.environ.get("DEMO_API_KEY")

if key is None:
    print("DEMO_API_KEY is not set. Run: export DEMO_API_KEY='your-key-here'")
    print("Falling back to a dummy value for demonstration.")
    os.environ["DEMO_API_KEY"] = "demo-key-0000-0000"
    key = os.environ["DEMO_API_KEY"]

print(f"Key loaded: {key[:8]}...{key[-4:]}")
```

Run this and observe the masked key prefix and suffix. The full value never appears in output — this habit prevents accidental exposure in log aggregators and CI consoles.

Now build the full client. This code calls `httpbin.org`, a free HTTP testing service that echoes your request back to you, so it runs without a real API key. The `bearer` endpoint validates that the `Authorization` header was sent correctly:

```python
import os
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
                f"Environment variable '{key_env_var}' is not set. "
                f"Export it before instantiating APIClient."
            )

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def get(self, path, params=None):
        url = f"{self.base_url}{path}"
        response = requests.get(
            url, headers=self._headers(), params=params, timeout=10
        )
        print(f"[{response.status_code}] GET {url}")

        if response.status_code == 401:
            raise AuthError(
                "Server rejected the API key. Verify the value and its scopes."
            )
        if response.status_code == 429:
            raise RateLimitError(
                "Rate limit exceeded. Back off before retrying."
            )
        if response.status_code >= 500:
            raise ServerError(
                f"Server returned {response.status_code}. Retry with backoff."
            )
        response.raise_for_status()
        return response.json()


os.environ.setdefault("DEMO_API_KEY", "demo-token-abc123")
client = APIClient("https://httpbin.org")
result = client.get("/bearer")
print(f"Server received token: {result['token']}")
```

The output confirms the full round-trip: the client read the key from the environment, injected it into the `Authorization` header, sent the request, and the server echoed it back. The same `_headers()` method runs on every `get()` call — there is no code path where a request goes out without authentication.

The error routing matters more than the happy path. When you run enrichment across 10,000 contacts at 2 AM, a 429 from Apollo means you need to slow down, not crash. A 401 means a key expired or was revoked, not that the data is bad. A 500 means the server had a transient failure, not that your request was wrong. Each error type gets its own exception so the calling code can respond differently — retry on 429, alert a human on 401, retry with backoff on 5xx.

For comparison, here is the same call using the Anthropic Python SDK. The SDK abstracts away the HTTP layer entirely — you never see the `Authorization` header, the base URL, or the JSON parsing. This is convenient for prototyping but makes debugging harder when something breaks, because you cannot inspect the raw request:

```python
import os
import anthropic

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-placeholder")

client = anthropic.Anthropic()

try:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=64,
        messages=[
            {"role": "user", "content": "Reply with exactly: AUTH_OK"}
        ],
    )
    print(response.content[0].text)
except anthropic.AuthenticationError as e:
    print(f"Auth failed: {e}")
except anthropic.RateLimitError as e:
    print(f"Rate limited: {e}")
```

If the key is a placeholder, the SDK raises `AuthenticationError` — which is the SDK's typed wrapper around the same 401 status code your raw client catches. The mechanism is identical. The SDK just hides the HTTP plumbing.

## Use It

Bearer-token HTTP authorization — the same shared-secret mechanism that authenticates calls to the Anthropic Messages API for LLM text generation — also authenticates every call inside a Clay enrichment waterfall that queries Apollo, Hunter, and People Data Labs for contact data. [CITATION NEEDED — concept: Clay enrichment waterfall provider call sequence] This is Cluster 1.2, TAM Refinement & ICP Scoring. The `APIClient` pattern you built above is the exact layer that sits between your contact list and those providers: read the key, inject the header, handle the status code.

The slice below simulates an enrichment lookup — the same request shape Clay sends to Apollo's people-search endpoint — using `httpbin.org` to echo the authenticated request so you can verify the header injection without a real key:

```python
import os, requests, time

os.environ.setdefault("APOLLO_API_KEY", "demo-apollo-key")

def enrich_contact(domain, title, max_retries=3):
    key = os.environ["APOLLO_API_KEY"]
    for attempt in range(max_retries):
        resp = requests.post(
            "https://httpbin.org/anything",
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json={"q_organization_domains": domain, "q_titles": title},
            timeout=10,
        )
        print(f"[{resp.status_code}] Attempt {attempt + 1}: {domain} / {title}")
        if resp.status_code == 200:
            body = resp.json()
            print(f"  Authorization sent: {body['headers'].get('Authorization', 'MISSING')[:20]}...")
            return body["json"]
        if resp.status_code == 429:
            wait = 2 ** attempt
            print(f"  Rate limited — backing off {wait}s")
            time.sleep(wait)
            continue
        return {"error": f"HTTP {resp.status_code}"}
    return {"error": "max retries exceeded"}

result = enrich_contact("anthropic.com", "Head of Sales")
print(f"Enrichment payload echoed: {result}")
```

Run this and the server echoes back the exact `Authorization` header your client injected, confirming the key never touched the URL, the query string, or the response body. Swap the `httpbin.org` URL for Apollo's real `/v1/mixed_people/search` endpoint and set a live `APOLLO_API_KEY` — the auth layer does not change.

## Exercises

**Exercise 1 (Easy):** Add a `retry_on_5xx` parameter to the `APIClient.get()` method. When `retry_on_5xx=True` and the server returns a 500-level status, sleep for `2 ** attempt` seconds and retry up to three times before raising `ServerError`. Test against `https://httpbin.org/status/500`, which always returns 500.

**Exercise 2 (Hard):** Build a `MultiProviderEnrichmentClient` that accepts a list of `(name, base_url, key_env_var)` tuples and implements a waterfall: query the first provider, and if it returns no result (empty `data` key or HTTP error), fall back to the next provider. Return the first non-empty result along with which provider produced it. Use `httpbin.org` endpoints that simulate different responses (`/anything` for success, `/status/404` for not-found) to test the fallback chain without real API keys.

## Key Terms

- **API Key** — A long string generated by the server and presented by the client on every request to prove identity. Treated as a shared secret; never encrypted in transit beyond HTTPS.
- **Bearer Token** — An authorization scheme (`Authorization: Bearer <token>`) where the server treats any holder of the token as authorized. The token "bears" the authorization.
- **Key-in-Query** — Authentication by appending the API key as a URL parameter (`?api_key=...`). Simpler to test but exposes the key in server and proxy logs.
- **OAuth 2.0 Client Credentials** — A two-step flow where a client exchanges a static ID and secret for a short-lived access token, then uses that token for subsequent requests. Used by Salesforce and HubSpot for server-to-server integrations.
- **Exponential Backoff** — A retry strategy where wait time doubles after each failed attempt (1s, 2s, 4s, 8s), preventing thundering-herd retries against a rate-limited server.
- **Environment Variable** — A key-value pair stored in the process environment (`os.environ`) rather than in source code, loaded at runtime so secrets never enter version control.

## Sources

- Anthropic API authentication documentation: <https://docs.anthropic.com/en/api/getting-started-auth>
- Hunter API documentation (key-in-query pattern): <https://hunter.io/api-documentation>
- Apollo API documentation (key-in-header pattern): <https://docs.apollo.io/reference/overview>
- HTTP Authentication Scheme Registry (Bearer): <https://www.rfc-editor.org/rfc/rfc6750>
- OAuth 2.0 Client Credentials Grant: <https://www.rfc-editor.org/rfc/rfc6749#section-4.4>
- [CITATION NEEDED — concept: Clay enrichment waterfall provider call sequence and internal auth handling]
- [CITATION NEEDED — concept: HubSpot server-to-server OAuth client credentials flow]
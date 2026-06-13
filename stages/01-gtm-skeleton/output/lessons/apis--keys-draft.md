# APIs & Keys

---

## Beat 1: Hook

Every enrichment waterfall, every CRM sync, every automated outreach sequence runs on API calls — and every API call requires authentication. A mismanaged key leaks in a GitHub commit, burns through rate limits, or silently fails at 2 AM. This lesson builds the authentication layer underneath every GTM automation stack.

---

## Beat 2: Concept

**API keys as shared secrets.** The server issues a string. The client presents it on every request. The server compares. That is the entire mechanism — no encryption of the key itself, just transport security (HTTPS) and obscurity.

**Three patterns you will see in GTM tooling:**
1. **Key-in-header** — `Authorization: Bearer sk-...` or `X-API-Key: ...`. Stateless. Every request carries the credential.
2. **Key-in-query** — `?api_key=...`. Common in older or simpler APIs. Visible in logs and browser history.
3. **OAuth 2.0 client credentials** — exchange a client ID + secret for a time-limited access token, then use the token. More moving parts, but the secret transmits less often.

**Rotation and scoping.** A key with `read:contacts` scope cannot delete records. A key with a 90-day expiry forces rotation. These are the two levers that limit blast radius.

**Environment variables.** Keys live in `ENV`, not in source files. The runtime injects them. Code references `os.environ["API_KEY"]`. Git never sees the value.

---

## Beat 3: Use It

**GTM Redirect → Zone 1 Enrichment (Clay waterfall, Apollo, Hunter, People Data Labs).**

Call a real enrichment API with a managed key. The practitioner stores the key in an environment variable, constructs an authenticated HTTP request, parses the JSON response, and prints the retrieved contact data. This is the exact call pattern Clay makes when it runs an enrichment waterfall — same endpoint, same auth header, same response shape.

Exercise hooks:
- **Easy:** Query a public API (no key required) to confirm request/response cycle. Print status code and one field from the response body.
- **Medium:** Store a demo API key in an environment variable, call a free-tier enrichment endpoint, and print the returned company name and domain.
- **Hard:** Implement a retry wrapper that catches 401/429 errors, sleeps on rate limit, and raises a custom exception on auth failure — then run it against a live endpoint.

---

## Beat 4: Build It

Build a reusable `APIClient` class that:
1. Reads the key from an environment variable at instantiation.
2. Injects the key into the `Authorization` header on every request.
3. Handles non-200 responses with specific error types (`AuthError`, `RateLimitError`, `ServerError`).
4. Logs the URL and status code of every call (without logging the key).

Exercise hooks:
- **Easy:** Instantiate the client with a missing env var and confirm it raises a clear error, not a cryptic `NoneType` failure.
- **Medium:** Make a live call to a free API using the client. Print the parsed response.
- **Hard:** Add exponential backoff to the client. Prove it works by calling an endpoint that returns 429 on the first two tries and 200 on the third (mock the endpoint locally).

---

## Beat 5: Extend It

**Key rotation without downtime.** Two keys exist simultaneously (old and new). The client tries the primary key; on 401, it falls back to the secondary. This is how production systems rotate — you don't revoke the old key until the new one is confirmed working.

**Scoped parallel calls.** With two keys (different rate-limit pools), run concurrent requests and measure throughput vs. a single key. The rate limit is per-key, so parallelism is a function of key count.

Exercise hooks:
- **Medium:** Simulate a rotation. Load both keys, make a call with each, print success/failure, then "revoke" the old key by removing it from env and re-running.
- **Hard:** Write a script that fires 10 concurrent requests across two API keys and prints per-key response times. Demonstrate that the rate limit doubles.

---

## Beat 6: Ship It

**GTM Redirect → Foundational for Zones 1–3 (Enrichment, Outreach, Analytics).**

Ship a `.env` file convention, a `requirements.txt` with `python-dotenv` and `httpx`, and the `APIClient` class in a standalone `api_client.py` module. The practitioner can import this module in any future automation — Clay webhook handlers, CRM sync scripts, outbound sequence triggers — and rely on consistent auth, error handling, and logging.

Exercise hooks:
- **Easy:** Create the `.env` file, load it with `dotenv`, print a masked version of the key (first 4 chars + `****`).
- **Medium:** Import `APIClient` from the module into a separate script. Call a live API. Print results.
- **Hard:** Write a smoke-test script (`test_api_client.py`) that runs all three error-path scenarios (missing key, bad key, rate limit) and exits 0 only if all pass. Run it from the command line and confirm the exit code.

---

## Learning Objectives

1. **Configure** API authentication using environment variables and the `Authorization` header.
2. **Implement** a reusable HTTP client that handles 401, 429, and 5xx responses with typed exceptions.
3. **Execute** an authenticated request against a live enrichment API and extract structured data from the response.
4. **Diagnose** common failure modes (missing key, expired key, rate limit) from HTTP status codes and error bodies.
5. **Apply** key rotation and dual-key patterns to maintain uptime during credential changes.
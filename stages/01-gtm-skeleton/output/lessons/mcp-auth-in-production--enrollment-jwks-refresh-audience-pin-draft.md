# MCP Auth in Production — Enrollment, JWKS Refresh, Audience-Pinned Tokens

---

## Beat 1: Hook

You built an MCP server on localhost with no auth. It worked. Now you're deploying to a shared environment and every client is `anonymous`. The protocol spec says OAuth 2.1 — but the spec doesn't tell you why your tokens work in Postman and fail in Claude Desktop, or why rotating a signing key invalidates every active session. This lesson covers the three mechanisms that break first in production: enrollment discovery, JWKS key rotation, and audience validation.

---

## Beat 2: Concept

**Enrollment** is the process by which an MCP client discovers what auth a server requires and obtains the credentials to satisfy it. The server advertises its requirements via `.well-known/oauth-authorization-server` metadata. The client reads that metadata, initiates the appropriate flow, and presents the resulting token on every request.

**JWKS (JSON Web Key Set)** is the mechanism that decouples token verification from key distribution. The server publishes its public keys at a known URL. The client fetches keys on demand, caches them by `kid`, and re-fetches when it encounters an unknown `kid`. This is what allows key rotation without redeploying clients.

**Audience pinning** prevents token confusion — an attack where a token issued for one resource is accepted by another. The `aud` claim in a JWT names the intended recipient. A production MCP server must reject any token where `aud` does not match its own identifier.

These three mechanisms form a chain: enrollment gets the token, audience pinning scopes it, JWKS lets you rotate the signing key without breaking either.

---

## Beat 3: Demonstration

Build a minimal MCP auth flow end-to-end using only standard libraries:

1. **Generate an RSA keypair and publish a JWKS endpoint** — Node.js `crypto` module, no external dependencies. Server exposes `GET /.well-known/jwks.json`.

2. **Issue an audience-pinned JWT** — Token includes `aud: "mcp://my-server"`, `iss`, `sub`, `kid` matching the JWKS entry. Observable output: decoded token payload printed to console.

3. **Validate the token server-side** — Fetch JWKS, match `kid`, verify signature, check `aud`. Observable output: `VALID` or `INVALID` with reason.

4. **Rotate the key** — Generate new keypair, publish both keys in JWKS, sign new tokens with new `kid`. Old tokens still validate during overlap window. Observable output: both old and new tokens validate.

Exercise hooks:
- **Easy:** Modify the `aud` claim to a wrong value, observe the rejection.
- **Medium:** Add a `kid` not present in JWKS, observe the fetch-and-fail behavior.
- **Hard:** Implement the full `.well-known/oauth-authorization-server` discovery document and write a client that reads it to determine the correct token endpoint.

---

## Beat 4: Use It

**GTM Redirect:** This is foundational for **Zone 2 — Agent Infrastructure**. Specifically: any MCP server that wraps a GTM tool (enrichment API, CRM write endpoint, scoring model) must authenticate callers before executing actions that mutate pipeline data.

[CITATION NEEDED — concept: MCP server auth requirements for production GTM tool integrations]

The mechanism applies directly to Clay waterfalls and similar enrichment flows: when an MCP agent calls an enrichment tool, the token's `aud` claim identifies which tool server is authorized, and JWKS rotation lets you cycle credentials without breaking active enrichment workflows.

Exercise hooks:
- **Medium:** Write an MCP server that exposes a `POST /enrich` endpoint, rejects tokens without `aud: "mcp://enrich-service"`, and logs the `sub` claim for audit.
- **Hard:** Simulate two MCP servers (enrichment and CRM write) and demonstrate that a token scoped to one is rejected by the other.

---

## Beat 5: Ship It

Production deployment checklist — each item is a mechanism, not a recommendation:

1. **JWKS cache TTL** — Set `Cache-Control` headers on the JWKS endpoint. Clients must respect the TTL before re-fetching. Without this, key rotation triggers a thundering herd.

2. **Key overlap window** — When rotating, publish both old and new keys in JWKS for at least 2x the token lifetime. Remove the old key only after all tokens signed with it have expired.

3. **Audience string format** — Use `mcp://<server-identifier>` as documented in the MCP spec. Do not reuse URLs or generic strings. The `aud` claim is a comparison value, not decorative.

4. **Enrollment metadata** — Deploy `.well-known/oauth-authorization-server` with correct `token_endpoint`, `jwks_uri`, and `scopes_supported`. Clients that can't discover this metadata fall back to... nothing. They fail silently.

5. **Token lifetime** — Short-lived tokens (5–15 minutes) with refresh reduce the blast radius of a leaked token. This is not a JWKS concern — it's an enrollment concern.

Exercise hooks:
- **Medium:** Deploy a JWKS endpoint with `Cache-Control: max-age=3600`, rotate the key after 30 seconds, and demonstrate that the client continues using the cached key until TTL expires.
- **Hard:** Implement automated key rotation: generate a new keypair every hour, maintain a 2-hour overlap window, log which `kid` is active vs. deprecated.

---

## Beat 6: Debug It

Failure modes and their signatures:

| Symptom | Cause | Diagnostic |
|---------|-------|------------|
| `401` on every request from Claude Desktop, works in curl | Missing or malformed `.well-known` metadata | Fetch `.well-known/oauth-authorization-server` directly, verify `jwks_uri` is reachable |
| `invalid_token` after key rotation | Client cached old JWKS, new `kid` not in cache | Check JWKS endpoint returns both keys, check client logs for `kid` mismatch |
| Token accepted by wrong server | No `aud` validation on server | Decode token, check `aud` claim, check server validation logic |
| `401` only on first request after deploy | JWKS endpoint not reachable at server startup | Hit JWKS URL from outside the network, check DNS and TLS |

**Debugging commands:**
```bash
# Decode a JWT without verifying (inspect claims)
echo $TOKEN | cut -d. -f2 | base64 -d 2>/dev/null | python3 -m json.tool

# Fetch and inspect JWKS
curl -s https://your-server/.well-known/jwks.json | python3 -m json.tool

# Check .well-known metadata
curl -s https://your-server/.well-known/oauth-authorization-server | python3 -m json.tool
```

Exercise hook:
- **Medium:** Given three failing scenarios (expired token, wrong `aud`, missing `kid`), write a diagnostic script that decodes the token, fetches JWKS, and prints the specific failure reason with remediation.

---

## Learning Objectives

1. **Implement** a JWKS endpoint that publishes RSA public keys and supports multi-key rotation with overlap windows.
2. **Configure** audience-pinned JWT validation that rejects tokens scoped to the wrong MCP server.
3. **Deploy** `.well-known/oauth-authorization-server` metadata that enables automated client enrollment.
4. **Diagnose** the three most common MCP auth failures in production: metadata discovery failures, stale JWKS caches, and audience mismatches.
5. **Compare** token lifetime strategies and their impact on key rotation scheduling.
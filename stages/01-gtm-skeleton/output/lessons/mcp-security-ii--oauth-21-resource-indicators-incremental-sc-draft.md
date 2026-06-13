# MCP Security II — OAuth 2.1, Resource Indicators, Incremental Scopes

## Hook

A compromised MCP server token with overly broad scopes becomes a lateral movement vector across every resource server your client touches. OAuth 2.1 removes the footguns (implicit grant, loose audience validation); resource indicators (RFC 8707) and incremental scoping constrain blast radius. This lesson covers the mechanism: how a client proves identity to an authorization server, receives scoped tokens bound to specific resources, and expands permissions only when required.

## Concept

Three mechanisms, one threat model: **token exfiltration + misuse**.

1. **OAuth 2.1 consolidation**: PKCE mandatory for all public clients, implicit grant removed, exact redirect URI matching, refresh token rotation. The spec tightens what was previously optional guidance into required constraints.

2. **Resource indicators (RFC 8707)**: The `resource` parameter in authorization and token requests binds the resulting access token to a specific resource server URI. A token issued for `https://crm.internal` is rejected by `https://enrichment.internal`. Mechanism: authorization server encodes the audience claim; resource server validates it. Prevents confused deputy attacks where MCP server A uses a token meant for MCP server B.

3. **Incremental authorization**: Client requests minimal scopes initially (`read:accounts`), then requests additional scopes (`write:accounts`, `read:contacts`) only when the operation requires them. Mechanism: subsequent authorization requests include previously granted scopes plus new ones; the authorization server returns a consolidated token. Reduces exposure window and aligns with least-privilege enforcement.

Key distinction: resource indicators restrict *where* a token is valid; scopes restrict *what* a token can do; incremental authorization restricts *when* expanded permissions exist.

- *Exercise hook (easy)*: Given three MCP server URLs and a set of required scopes, identify which combination of `resource` and `scope` parameters should appear in each authorization request.
- *Exercise hook (medium)*: Trace a token exchange where a client requests `read:leads` initially, then escalates to `write:leads` — diagram which HTTP requests change and which tokens are re-issued.
- *Exercise hook (hard)*: Identify the confused deputy vulnerability in a provided sequence where resource indicators are absent but scope validation is present; explain why scope-only validation is insufficient.

## Demo

Build a minimal authorization server mock and two resource servers. Client authorizes with PKCE, requests token with `resource` parameter targeting resource server A. Attempt to use that token against resource server B — observe rejection. Then request incremental scope expansion and observe the new token's claims.

Observable output: each HTTP request/response printed to terminal with headers, token claims decoded (base64 payload, no library required), and clear accept/reject outcomes.

```python
import hashlib, base64, secrets, json, time

def generate_pkce_pair():
    verifier = secrets.token_urlsafe(32)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()
    return verifier, challenge

def mock_authz_server(resource_uri, scopes_requested, previously_granted=None):
    all_scopes = list(set((previously_granted or []) + scopes_requested))
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).decode().rstrip('=')
    payload_data = {
        "scope": ' '.join(all_scopes),
        "aud": resource_uri,
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600
    }
    payload = base64.urlsafe_b64encode(json.dumps(payload_data).encode()).decode().rstrip('=')
    token = f"{header}.{payload}.mocksig"
    return token, payload_data

def resource_server_validate(token, expected_resource, required_scope):
    parts = token.split('.')
    padded = parts[1] + '=' * (4 - len(parts[1]) % 4)
    claims = json.loads(base64.urlsafe_b64decode(padded))
    aud_match = claims['aud'] == expected_resource
    scope_match = required_scope in claims['scope'].split(' ')
    return {'accepted': aud_match and scope_match, 'claims': claims, 'aud_match': aud_match, 'scope_match': scope_match}

verifier, challenge = generate_pkce_pair()
print(f"PKCE verifier: {verifier}")
print(f"PKCE challenge: {challenge}")
print()

token_a, claims_a = mock_authz_server("https://crm.internal", ["read:accounts", "read:leads"])
print(f"Token A (for crm.internal): {token_a}")
print(f"Token A claims: {json.dumps(claims_a, indent=2)}")
print()

result_wrong_resource = resource_server_validate(token_a, "https://enrichment.internal", "read:accounts")
print(f"Token A -> enrichment.internal: accepted={result_wrong_resource['accepted']} (aud_match={result_wrong_resource['aud_match']})")
print()

result_correct_resource = resource_server_validate(token_a, "https://crm.internal", "read:leads")
print(f"Token A -> crm.internal (read:leads): accepted={result_correct_resource['accepted']}")
print()

result_missing_scope = resource_server_validate(token_a, "https://crm.internal", "write:leads")
print(f"Token A -> crm.internal (write:leads): accepted={result_missing_scope['accepted']} (scope_match={result_missing_scope['scope_match']})")
print()

token_a_incremented, claims_incremented = mock_authz_server("https://crm.internal", ["write:leads"], previously_granted=["read:accounts", "read:leads"])
print(f"Token A incremented claims: {json.dumps(claims_incremented, indent=2)}")

result_after_increment = resource_server_validate(token_a_incremented, "https://crm.internal", "write:leads")
print(f"Incremented token -> crm.internal (write:leads): accepted={result_after_increment['accepted']}")
```

- *Exercise hook (easy)*: Run the code. Modify the `resource_uri` parameter and confirm audience validation fails for a different URI.
- *Exercise hook (medium)*: Add a third resource server and a token that is valid for two audiences. Observe what breaks in the current validator and patch the `aud` check to support list-valued audiences.
- *Exercise hook (hard)*: Implement refresh token rotation: issue a refresh token alongside the access token, consume it once to get a new pair, and reject reuse of the original refresh token.

## Use It

**GTM cluster connection**: This maps to Zone 3 (Enrichment & Scoring) and Zone 4 (Multi-Channel Outreach) in the GTM topic map. When an MCP client orchestrates a Clay waterfall — sequentially calling Apollo, Hunter, and Clearbit enrichment — each third-party API is a distinct resource server. Resource indicators prevent a token issued for Apollo from being replayed against Clearbit. Incremental scopes map to progressive access: initial waterfall needs `people:search`, later stages need `people:email` — request only what the current waterfall step requires.

Concrete scenario: a GTM engineer configures an MCP client that calls an enrichment server (resource: `https://enrichment.internal`, scope: `people:search`) and a CRM server (resource: `https://crm.internal`, scope: `contacts:write`). A token exfiltrated from the enrichment server's logs cannot create contacts in the CRM because the audience claim does not match. [CITATION NEEDED — concept: Clay waterfall OAuth scope configuration per enrichment provider]

- *Exercise hook (medium)*: Map the scopes and resource URIs for a three-step enrichment waterfall (company lookup → contact lookup → email verification) and identify at which step each scope is incrementally requested.

## Ship It

Production checklist for deploying OAuth 2.1 + resource indicators with MCP servers:

1. **Authorization server configuration**: Enforce PKCE for all clients (no exceptions for "confidential" clients — the marginal complexity saving isn't worth the footgun). Require exact redirect URI match. Enable refresh token rotation with one-time use enforcement.

2. **Resource server validation**: On every inbound request, validate `aud` claim against the server's canonical URI *before* checking scopes. Reject tokens with missing or mismatched `aud` with `403` not `401` (the token is valid, just not for this resource).

3. **Scope storage and expansion**: Store currently granted scopes server-side (not just in the token). When incremental authorization occurs, merge and re-issue. Log scope escalation events for audit.

4. **Token metadata for MCP**: Include `mcp_server_id` as a custom claim to disambiguate multiple MCP servers behind the same resource URI. Validate on receipt.

5. **Rotation and revocation**: Rotate refresh tokens on every use. Revoke access tokens on scope reduction. Expose a `.well-known/oauth-authorization-server` endpoint for dynamic client discovery.

- *Exercise hook (hard)*: Write a middleware function that extracts the bearer token, validates `aud` and `scope` against a configurable resource URI and required scope list, and returns `403` with a structured error body on mismatch. Include observable logging that prints the decision path.

## Review

Three mechanisms, one principle: least privilege enforced at the token level.

- OAuth 2.1 removed ambiguity (PKCE mandatory, implicit grant gone, redirect URI exact match)
- Resource indicators bind tokens to resource servers (audience restriction prevents confused deputy)
- Incremental authorization minimizes standing permissions (request wide scopes only when needed)

For assessment: expect to trace a token lifecycle from authorization request through resource server validation, identify where a confused deputy attack would succeed without resource indicators, and construct the correct `resource` + `scope` parameters for a multi-server MCP deployment.

- *Exercise hook (easy)*: Write the three differences between OAuth 2.0 best practices and OAuth 2.1 mandatory constraints.
- *Exercise hook (medium)*: Given a token payload, determine which resource servers will accept it and which operations are permitted — explain your reasoning for each accept/reject decision.
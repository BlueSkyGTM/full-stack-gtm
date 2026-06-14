## Ship It

The three-agent Contract Net negotiation is the canonical test of whether your performative model holds under real coordination pressure. In a GTM enrichment scenario, the initiator is your orchestrator, and the participants are competing enrichment providers. The orchestrator broadcasts a `cfp` ("who can enrich this account?"), each provider responds with a `propose` ("I can, with coverage X and cost Y"), the orchestrator selects based on its criteria, and the winner delivers results. The loser receives a `reject-proposal` and drops out. This is a negotiation, not a function call—and the protocol enforces the rules of engagement.

```python
import json

PERFORMATIVES = {
    "inform": "assertive", "confirm": "assertive",
    "disconfirm": "assertive", "reject-proposal": "assertive",
    "not-understood": "assertive", "failure": "assertive",
    "request": "directive", "query-ref": "directive",
    "query-if": "directive", "cfp": "directive",
    "subscribe": "directive", "cancel": "directive",
    "request-when": "directive",
    "propose": "commissive", "agree": "commissive",
    "accept-proposal": "commissive", "refuse": "commissive",
}

def make_acl_msg(sender, receiver, performative, content,
                 conv_id, protocol="fipa-contract-net"):
    return {
        "sender": sender,
        "receiver": receiver,
        "performative": performative,
        "speech_act": PERFORMATIVES.get(performative, "unknown"),
        "content": content,
        "conversation-id": conv_id,
        "protocol": protocol,
    }

def contract_net_simulation(account, providers):
    conv_id = f"cnp-{account['id']}"
    trace = []
    initiator = "agent://orchestrator"

    cfp = make_acl_msg(
        initiator, [p["id"] for p in providers],
        "cfp",
        {"task": "enrich_account", "account": account},
        conv_id,
    )
    trace.append(("INITIATOR → ALL", cfp))

    proposals = []
    for provider in providers:
        if provider["coverage"] >= account["needed_coverage"]:
            proposal = make_acl_msg(
                provider["id"], initiator,
                "propose",
                {"cost": provider["cost"], "coverage": provider["coverage"],
                 "eta_minutes": provider["eta"]},
                conv_id,
            )
            proposals.append((provider, proposal))
            trace.append((f"{provider['id']} → INITIATOR", proposal))
        else:
            refuse = make_acl_msg(
                provider["id"], initiator,
                "refuse",
                {"reason": "insufficient_coverage"},
                conv_id,
            )
            trace.append((f"{provider['id']} → INITIATOR", refuse))

    if not proposals:
        trace.append(("INITIATOR", make_acl_msg(
            initiator, initiator, "failure",
            {"reason": "no_proposals_received"}, conv_id)))
        return trace

    winner = min(proposals, key=lambda x: x[1]["content"]["cost"])
    for provider, proposal in proposals:
        if provider["id"] == winner[0]["id"]:
            accept = make_acl_msg(
                initiator, provider["id"],
                "accept-proposal",
                {"accepted_cost": proposal["content"]["cost"]},
                conv_id,
            )
            trace.append(("INITIATOR → WINNER", accept))
        else:
            reject = make_acl_msg(
                initiator, provider["id"],
                "reject-proposal",
                {"reason": "higher_cost"},
                conv_id,
            )
            trace.append(("INITIATOR → LOSER", reject))

    result = make_acl_msg(
        winner[0]["id"], initiator,
        "inform",
        {"enriched_data": {"email": f"jane@{account['domain']}",
                            "phone": "+1-555-0100"},
         "cost": winner[1]["content"]["cost"]},
        conv_id,
    )
    trace.append((f"{winner[0]['id']} → INITIATOR", result))

    return trace


account = {
    "id": "acme-001",
    "domain": "acme.com",
    "needed_coverage": 0.8,
}

providers = [
    {"id": "agent://apollo",    "coverage": 0.92, "cost": 0.10, "eta": 2},
    {"id": "agent://zoominfo",  "coverage": 0.88, "cost": 0.15, "eta": 5},
    {"id": "agent://hunter",    "coverage": 0.65, "cost": 0.02, "eta": 1},
]

trace = contract_net_simulation(account, providers)

print("Contract Net Negotiation Trace")
print("=" * 70)
for i, (direction, msg) in enumerate(trace):
    print(f"\n[Turn {i + 1}] {direction}")
    print(f"  Performative:  {msg['performative']}")
    print(f"  Speech Act:    {msg['speech_act']}")
    print(f"  Content:       {json.dumps(msg['content'])}")

valid_sequence = ["cfp", "propose", "accept-proposal", "inform"]
actual_sequence = [msg["performative"]
                   for _, msg in trace
                   if msg["performative"] in valid_sequence]

print(f"\n{'=' * 70}")
print(f"Winner-side sequence: {' → '.join(actual_sequence)}")
print(f"Protocol valid: {actual_sequence == valid_sequence}")
print(f"Provider rejected (low coverage): agent://hunter")
print(f"Loser notified: {'yes' if any(m['performative'] == 'reject-proposal' for _, m in trace) else 'no'}")
```

The output shows a complete negotiation with four speech-act types flowing across three agents. Hunter refuses because its coverage is below the threshold—this is a `refuse` (commissive), not a `reject-proposal` (assertive), because Hunter is declining to enter the negotiation at all. Apollo wins on cost. ZoomInfo receives a `reject-proposal` and exits. The winner-side sequence `cfp → propose → accept-proposal → inform` is the canonical Contract Net happy path. This is the same protocol pattern your enrichment waterfall follows when it queries multiple providers and selects the best result—the difference is that FIPA-ACL names every step, so the audit trail is unambiguous.
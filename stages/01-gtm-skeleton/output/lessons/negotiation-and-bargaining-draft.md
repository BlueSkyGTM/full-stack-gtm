# Negotiation and Bargaining

---

## Beat 1: Hook

Two agents, conflicting utility functions, shared resource. This is the core constraint in any multi-agent system where both parties want different outcomes from the same transaction. Every pricing conversation, vendor contract, and SLA negotiation is this pattern.

---

## Beat 2: Concept

Introduce the formal model: alternating offers, utility curves, reservation prices, and the Nash equilibrium that emerges when both agents have complete information. Contrast with the incomplete-information variant (real-world condition) where signaling and screening replace direct revelation.

Key terms: BATNA (Best Alternative to Negotiated Agreement), zone of possible agreement (ZOPA), Pareto frontier, and how agents compute strategy when the opponent's utility function is partially observable.

---

## Beat 3: Mechanism

Walk through the algorithmic approaches:

1. **Nash Bargaining Solution** — maximize the product of surplus utilities. Closed-form when both utility functions are known.
2. **Rubinstein Alternating Offers** — infinite-horizon game with discount factors. Show why the first-mover advantage shrinks as patience (discount factor) converges between agents.
3. **Automated Negotiation via LLM Agents** — prompt-driven agents with encoded preferences, using structured output to emit offers and evaluate counteroffers.

Show the math for Rubinstein: if player 1 discounts at δ₁ and player 2 at δ₂, the subgame-perfect equilibrium split is computable. Then show how to implement this as a negotiation loop in code.

---

## Beat 4: Use It

**GTM Redirect:** Deal Desk Automation — specifically, automating pricing negotiation for enterprise sales where the buyer's willingness-to-pay is unknown and must be inferred through offer-response sequences.

Map negotiation concepts to GTM:
- BATNA = the buyer's next-best vendor option
- ZOPA = the overlap between your floor price and their ceiling
- Discount factor = how urgently the buyer needs the solution this quarter vs. next quarter

Build a simulation: two LLM agents negotiating a SaaS contract price. One agent encodes the vendor's cost floor and margin target. The other encodes the buyer's budget ceiling and feature requirements. The loop runs alternating offers until agreement or walk-away.

**Exercise (Easy):** Configure a single-issue negotiation (price only) with known reservation prices for both agents. Confirm the agents converge on a price within the ZOPA.

**Exercise (Medium):** Add a second issue (contract length) and show how log-rolling — trading concessions across issues — produces Pareto-superior outcomes compared to single-issue negotiation.

**Exercise (Hard):** Introduce incomplete information: the buyer agent does not reveal its true reservation price. Implement a Bayesian update mechanism where the vendor agent revises its belief about the buyer's ceiling after each rejected offer.

---

## Beat 5: Ship It

Implement a production-grade negotiation framework:

```python
import json
from dataclasses import dataclass, field
from typing import Optional, Callable

@dataclass
class Offer:
    price: float
    contract_length_months: int
    features_included: list[str] = field(default_factory=list)
    round_number: int = 0

@dataclass
class NegotiationState:
    vendor_floor: float
    buyer_ceiling: float
    vendor_discount_factor: float
    buyer_discount_factor: float
    max_rounds: int
    history: list[dict] = field(default_factory=list)
    agreement: Optional[dict] = None

def rubinstein_split(state: NegotiationState) -> dict:
    d1 = state.vendor_discount_factor
    d2 = state.buyer_discount_factor
    vendor_share = (1 - d2) / (1 - (d1 * d2))
    buyer_share = 1 - vendor_share
    surplus = state.buyer_ceiling - state.vendor_floor
    return {
        "vendor_surplus": round(surplus * vendor_share, 2),
        "buyer_surplus": round(surplus * buyer_share, 2),
        "predicted_price": round(state.vendor_floor + surplus * vendor_share, 2),
        "mechanism": "rubinstein_alternating_offers"
    }

def evaluate_offer(offer: Offer, state: NegotiationState, role: str) -> dict:
    if role == "vendor":
        acceptable = offer.price >= state.vendor_floor
        utility = offer.price - state.vendor_floor
    else:
        acceptable = offer.price <= state.buyer_ceiling
        utility = state.buyer_ceiling - offer.price
    return {"acceptable": acceptable, "utility": round(utility, 2)}

state = NegotiationState(
    vendor_floor=50000,
    buyer_ceiling=80000,
    vendor_discount_factor=0.9,
    buyer_discount_factor=0.85,
    max_rounds=10
)

equilibrium = rubinstein_split(state)
print("Rubinstein Equilibrium:")
print(json.dumps(equilibrium, indent=2))

test_offer = Offer(price=65000, contract_length_months=12, round_number=1)
vendor_eval = evaluate_offer(test_offer, state, "vendor")
buyer_eval = evaluate_offer(test_offer, state, "buyer")
print(f"\nOffer: ${test_offer.price}")
print(f"Vendor evaluation: {vendor_eval}")
print(f"Buyer evaluation: {buyer_eval}")
print(f"ZOPA exists: {state.vendor_floor < state.buyer_ceiling}")
print(f"ZOPA range: ${state.vendor_floor} - ${state.buyer_ceiling}")
```

This code runs unmodified and prints the equilibrium prediction, offer evaluation, and ZOPA confirmation.

**Exercise (Medium):** Extend `NegotiationState` to track the full alternating-offer sequence. Implement a loop where each side proposes, the other evaluates, and the loop terminates on agreement or `max_rounds` exhaustion. Print the full offer history at the end.

---

## Beat 6: Evaluate It

Metrics for negotiation quality:

1. **Pareto efficiency** — no alternative agreement exists that makes one agent better off without making the other worse off. Check by enumerating all feasible agreements and plotting the utility frontier.
2. **Distance from Nash Bargaining Solution** — how far did the actual agreement deviate from the theoretical optimum? Larger deviation indicates either irrational agent behavior or information asymmetry costs.
3. **Convergence rate** — number of rounds to agreement. Rubinstein predicts immediate acceptance in the complete-information case. Observe how many extra rounds incomplete information adds.
4. **Walk-away rate** — percentage of simulations ending without agreement. High walk-away rate with a known ZOPA indicates poor offer generation or miscalibrated agents.

Implement a benchmarking harness that runs 100 negotiation simulations with randomized reservation prices and discount factors, then reports the four metrics above.

**Exercise (Hard):** Run a parameter sweep over `buyer_discount_factor` from 0.5 (impatient) to 0.99 (patient). Plot how the walk-away rate and average rounds-to-agreement change. Identify the discount factor threshold where the vendor's first offer acceptance rate drops below 50%.

---

## GTM Redirect Summary

- **Use It** → Deal Desk Automation cluster: pricing negotiation, BATNA estimation, ZOPA detection
- **Ship It** → Production negotiation loop for automated contract proposal evaluation
- **Evaluate It** → Negotiation outcome quality metrics applicable to any GTM scenario involving price/terms discussion

If the negotiation concept does not cleanly map to a specific GTM workflow (e.g., multi-party negotiation with more than two agents), the redirect defaults to: **foundational for Zone 02 (Prospect → Opportunity) and Zone 03 (Opportunity → Close)**.
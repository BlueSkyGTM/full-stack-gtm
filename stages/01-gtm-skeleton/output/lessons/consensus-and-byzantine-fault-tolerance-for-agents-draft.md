# Consensus and Byzantine Fault Tolerance for Agents

## Hook
When three agents score a lead and one returns garbage, majority vote works. When that agent lies strategically about *other agents' votes*, you have a Byzantine problem. This lesson covers the algorithms that let distributed agent systems reach agreement even when some participants are malicious or arbitrarily faulty.

## Concept
The Byzantine Generals Problem: n generals must agree on a single action, but some may be traitors sending contradictory messages. Distinction between crash faults (node stops) and Byzantine faults (node behaves arbitrarily). Why agent networks face Byzantine risks: compromised API keys, poisoned model outputs, prompt-injected agents returning crafted responses to manipulate group decisions.

## Mechanism
Practical Byzantine Fault Tolerance (PBFT) — the algorithm Castro and Liskov published in 1999. Phases: pre-prepare, prepare, commit. Why 3f+1 nodes are needed to tolerate f Byzantine faults (proof sketch). How quorum intersection guarantees safety: any two quorums of size 2f+1 overlap in at least f+1 honest nodes. Compare to Raft (crash-fault only, simpler, 2f+1). Compare to Tendermint/BFT proof-of-stake (validator sets, voting power).

## Code
Python simulation of a 4-node PBFT vote: one honest proposer, two honest validators, one Byzantine validator sending conflicting `PREPARE` messages to different peers. Print each message exchange, show that honest nodes still reach commit despite the double-vote. Observable output: log of all phases and final committed value per node.

## Use It
Multi-agent enrichment pipelines in GTM. When you run parallel enrichment agents (firmographic, intent, technographic) and need to reconcile conflicting signals about an account, naive averaging hides Byzantine manipulation. Consensus-based signal aggregation: each agent signs its output, a quorum must agree before the signal enters the score. This is foundational for Zone 30 (Enrichment) and Zone 40 (Scoring) — specifically, multi-source signal reconciliation in Clay waterfalls where `n` data providers return conflicting firmographic data and you need a principled resolution, not a `COALESCE`. [CITATION NEEDED — concept: Clay waterfall consensus/reconciliation pattern for conflicting enrichment providers]

## Ship It
Implement a 4-node agent committee that scores accounts using PBFT. One node is injected with adversarial prompts that attempt to manipulate the committee's final score. Medium exercise: detect the Byzantine node by comparing commit logs across honest nodes. Hard exercise: implement view-change when the primary node is the Byzantine one — detect timeout, elect new primary, resume consensus. Output: before/after scores showing the system converges despite the attack.
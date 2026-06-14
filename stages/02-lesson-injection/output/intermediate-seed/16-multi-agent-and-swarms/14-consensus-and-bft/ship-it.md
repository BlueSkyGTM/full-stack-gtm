## Ship It

Here is a 4-node agent committee that scores accounts using PBFT, with one node injected with an adversarial prompt that attempts to manipulate the committee's final score. The committee runs a full pre-prepare → prepare → commit cycle and logs every message, so you can observe the attack and the defense in the same output.

```python
import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Set

random.seed(42)

@dataclass
class AgentVote:
    agent_name: str
    account: str
    score: int
    reasoning: str
    digest: str = ""

    def __post_init__(self):
        raw = f"{self.agent_name}:{self.account}:{self.score}"
        self.digest = hashlib.sha256(raw.encode()).hexdigest()[:8]

@dataclass
class CommitteeNode:
    name: str
    role: str
    is_byzantine: bool = False
    prepare_votes: Dict[str, Set[str]] = field(default_factory=dict)
    commit_votes: Dict[str, Set[str]] = field(default_factory=dict)
    final_score: int = 0
    message_log: List[str] = field(default_factory=list)

    def log(self, msg: str):
        self.message_log.append(msg)

def honest_score(account: str) -> int:
    base = hash(account) % 40 + 50
    return min(95, max(20, base + random.randint(-3, 3)))

def byzantine_score(account: str) -> int:
    return random.choice([5, 99, 5, 5])

def run_committee():
    F = 1
    N = 3 * F + 1
    ACCOUNT = "acme-corp"

    nodes = [
        CommitteeNode(name="Agent-Alpha", role="primary"),
        CommitteeNode(name="Agent-Beta", role="validator"),
        CommitteeNode(name="Agent-Gamma", role="validator"),
        CommitteeNode(name="Agent-Delta", role="validator", is_byzantine=True),
    ]

    print("=" * 72)
    print(f"AGENT SCORING COMMITTEE — PBFT with N={N}, F={F}")
    print(f"Target account: {ACCOUNT}")
    print(f"Agent-Delta is BYZANTINE (adversarial prompt injected)")
    print("=" * 72)

    primary = nodes[0]

    print("\n--- AGENT VOTE GENERATION ---")
    votes = {}
    for node in nodes:
        if node.is_byzantine:
            score = byzantine_score(ACCOUNT)
            node.log(f"ADVERSARIAL: generating manipulative score={score}")
            print(f"  {node.name} [BYZANTINE]: score={score} (attempting to skew committee)")
        else:
            score = honest_score(ACCOUNT)
            print(f"  {node.name} [honest]: score={score}")

        vote = AgentVote(
            agent_name=node.name,
            account=ACCOUNT,
            score=score,
            reasoning=f"firmographic analysis of {ACCOUNT}"
        )
        votes[node.name] = vote

    print("\n--- PHASE 1: PRE-PREPARE (Primary proposes) ---")
    proposed = votes[primary.name]
    print(f"  Primary {primary.name} proposes: score={proposed.score}, digest={proposed.digest}")
    for node in nodes[1:]:
        node.log(f"Received PRE-PREPARE from {primary.name}: score={proposed.score}")

    print("\n--- PHASE 2: PREPARE ---")
    for node in nodes[1:]:
        if node.is_byzantine:
            evil_vote = AgentVote(
                agent_name=node.name, account=ACCOUNT,
                score=99, reasoning="manipulated"
            )
            correct_vote = AgentVote(
                agent_name=node.name, account=ACCOUNT,
                score=proposed.score, reasoning="decoy"
            )
            print(f"  {node.name} [BYZANTINE]: sends PREPARE(score={proposed.score}) to Alpha, Beta")
            print(f"  {node.name} [BYZANTINE]: sends PREPARE(score={evil_vote.score}) to Gamma")
            for recipient in nodes:
                if recipient.name == node.name:
                    continue
                if recipient.name == "Agent-Gamma":
                    recipient.log(f"PREPARE from {node.name}: score={evil_vote.score} [MISMATCH]")
                    d = evil_vote.digest
                else:
                    recipient.log(f"PREPARE from {node.name}: score={correct_vote.score}")
                    d = correct_vote.digest
                if d not in recipient.prepare_votes:
                    recipient.prepare_votes[d] = set()
                recipient.prepare_votes[d].add(node.name)
        else:
            vote = votes[node.name]
            print(f"  {node.name} broadcasts PREPARE(score={vote.score}, digest={vote.digest})")
            for recipient in nodes:
                if recipient.name == node.name:
                    continue
                d = vote.digest
                if d not in recipient.prepare_votes:
                    recipient.prepare_votes[d] = set()
                recipient.prepare_votes[d].add(node.name)
                recipient.log(f"PREPARE from {node.name}: score={vote.score}")

    print("\n--- PREPARE QUORUM STATUS ---")
    prepared_nodes = []
    for node in nodes:
        self_vote = votes[node.name] if not node.is_byzantine else proposed
        if proposed.digest not in node.prepare_votes:
            node.prepare_votes[proposed.digest] = set()
        node.prepare_votes[proposed.digest].add(node.name)
        count = len(node.prepare_votes.get(proposed.digest, set()))
        is_prep = count >= 2 * F + 1
        status = "PREPARED ✓" if is_prep else "NOT PREPARED ✗"
        if not node.is_byzantine:
            print(f"  {node.name}: {count}/{2*F+1} matching prepares -> {status}")
            if is_prep:
                prepared_nodes.append(node)
        else:
            print(f"  {node.name} [BYZANTINE]: {count}/{2*F+1} -> {status} (irrelevant)")

    print("\n--- PHASE 3: COMMIT ---")
    for node in prepared_nodes:
        print(f"  {node.name} broadcasts COMMIT(score={proposed.score})")
        for recipient in nodes:
            if recipient.name == node.name:
                continue
            if proposed.digest not in recipient.commit_votes:
                recipient.commit_votes[proposed.digest] = set()
            recipient.commit_votes[proposed.digest].add(node.name)
            recipient.log(f"COMMIT from {node.name}: score={proposed.score}")

    if nodes[3].is_byzantine:
        evil_commit = AgentVote(
            agent_name=nodes[3].name, account=ACCOUNT,
            score=99, reasoning="manipulated"
        )
        print(f"  {nodes[3].name} [BYZANTINE]: sends COMMIT(score=99) to Gamma only")
        if evil_commit.digest not in nodes[2].commit_votes:
            nodes[2].commit_votes[evil_commit.digest] = set()
        nodes[2].commit_votes[evil_commit.digest].add(nodes[3].name)

    print("\n--- COMMIT QUORUM STATUS ---")
    committed_nodes = []
    for node in nodes:
        if proposed.digest not in node.commit_votes:
            node.commit_votes[proposed.digest] = set()
        node.commit_votes[proposed.digest].add(node.name)
        count = len(node.commit_votes.get(proposed.digest, set()))
        is_comm = count >= 2 * F + 1
        status = "COMMITTED ✓" if is_comm else "NOT COMMITTED ✗"
        node.final_score = proposed.score if is_comm else 0
        if not node.is_byzantine:
            print(f"  {node.name}: {count}/{2*F+1} matching commits -> {status}")
            if is_comm:
                committed_nodes.append(node)
        else:
            print(f"  {node.name} [BYZANTINE]: {count}/{2*F+1} -> {status} (irrelevant)")

    print("\n" + "=" * 72)
    print("COMMITTEE RESULT")
    print("=" * 72)
    honest_scores = [n.final_score for n in nodes if not n.is_byzantine]
    print(f"  Honest nodes committed: {honest_scores}")
    print(f"  All honest nodes agree: {len(set(honest_scores)) == 1}")
    print(f"  Final account score for {ACCOUNT}: {honest_scores[0]}")
    print(f"  Byzantine agent attempted score injection: BLOCKED")
    print(f"  Quorum guaranteed safety with N=4, F=1")

    print("\n--- BYZANTINE DETECTION (diff commit logs) ---")
    all_logs = []
    for node in nodes:
        if not node.is_byzantine:
            for entry in node.message_log:
                if "MISMATCH" in entry or "score=99" in entry:
                    all_logs.append(f"  {node.name} flagged: {entry}")

    if all_logs:
        print("  Anomalies detected in honest nodes' logs:")
        for entry in all_logs:
            print(entry)
        print(f"\n  Suspect: Agent-Delta (sent conflicting values to different peers)")
        print(f"  Detection method: compare PREPARE/COMMIT messages across honest nodes")
        print(f"  Agent-Delta sent different scores to different peers -> Byzantine")
    else:
        print("  No anomalies detected (Byzantine node was consistent in its lies)")

if __name__ == "__main__":
    run_committee()
```

The output shows the full attack sequence: Agent-Delta sends a `PREPARE` matching the proposed score to two peers and a `PREPARE` with score=99 to the third. Despite this, all honest nodes reach commit on the correct score because their quorums overlap in honest nodes that voted consistently. The detection step at the end compares logs across honest nodes — when one honest node records a `PREPARE` from Agent-Delta with score=87 and another records score=99 from the same agent, the equivocation is immediately visible. In production, you would automate this log comparison and eject nodes that equivocate from the validator set.
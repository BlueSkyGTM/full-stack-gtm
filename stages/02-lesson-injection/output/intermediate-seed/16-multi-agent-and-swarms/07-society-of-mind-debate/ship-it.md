## Ship It

Putting a debate system into a production GTM pipeline requires solving three engineering problems that the prototype above ignores: cost control, latency, and degradation detection.

Cost is the binding constraint. A 3-agent × 2-round debate on 5,000 enriched companies is 30,000 API calls. At Claude Sonnet pricing, that is real money before you send a single email. Two mitigations apply. First, run the debate only on companies that pass a cheap pre-filter — a single-agent score above some threshold that indicates "plausible fit, worth debating." This gates the expensive inference behind a cheap filter, the same pattern as an enrichment waterfall where you only call the expensive data provider for records that pass the cheap checks. Second, implement convergence-based early stopping. If all agents agree after round 1, skip round 2. For clear fits and clear non-fits, this cuts the call count in half. Only the ambiguous cases — the ones where agents disagree — get the full multi-round treatment, which is exactly where you want to spend the compute.

Latency is the second constraint. The debate is sequential by construction: round 2 requires round 1's outputs. But within each round, agent calls are independent and can be parallelized. Three agents in round 1 can fire as three concurrent API calls. This is a textbook distributed systems problem — parallel requests with rate limit backpressure, which maps to Zone 16 in the GTM curriculum. Your enrichment waterfall already handles this pattern for data providers; the same concurrency primitives apply to LLM calls. Use an async client, set a concurrency limit at or below your API tier's rate limit, and implement exponential backoff on 429 responses. The debate loop becomes: fire N agents concurrently per round, gather results, check convergence, either stop or proceed to the next round.

```python
import asyncio
import time

async def async_agent_call(agent_id, question, others, persona, round_num, client=None):
    await asyncio.sleep(0.05)
    return mock_llm(agent_id, question, others, persona, round_num)

async def async_run_debate(config: DebateConfig) -> list:
    transcript = []
    agents = config.personas if config.personas else [f"agent_{i}" for i in range(config.num_agents)]
    
    tasks = [async_agent_call(a, config.question, [], a, 1) for a in agents]
    results = await asyncio.gather(*tasks)
    round_responses = [
        AgentResponse(agents[i], results[i]["answer"], results[i]["reasoning"], 1)
        for i in range(len(agents))
    ]
    transcript.extend(round_responses)
    
    for round_num in range(2, config.num_rounds + 1):
        prev = [r for r in transcript if r.round_number == round_num - 1]
        tasks = []
        for agent in agents:
            others = [
                {"answer": r.answer, "reasoning": r.reasoning, "agent_id": r.agent_id}
                for r in prev if r.agent_id != agent
            ]
            tasks.append(async_agent_call(agent, config.question, others, agent, round_num))
        
        results = await asyncio.gather(*tasks)
        new_responses = [
            AgentResponse(agents[i], results[i]["answer"], results[i]["reasoning"], round_num)
            for i in range(len(agents))
        ]
        transcript.extend(new_responses)
        
        answers = [r.answer for r in new_responses]
        most_common = max(set(answers), key=answers.count)
        if answers.count(most_common) == len(answers):
            print(f"Converged at round {round_num} — stopping early")
            break
    
    return transcript

start = time.time()
config = DebateConfig(
    num_agents=3,
    num_rounds=3,
    question="Does Acme Cloud fit our ICP?",
    personas=["advocate", "skeptic", "analyst"],
)
transcript = asyncio.run(async_run_debate(config))
elapsed = time.time() - start
result = aggregate_debate(transcript, 3)
print(f"Completed {len(transcript)} agent calls in {elapsed:.2f}s")
print(json.dumps(result, indent=2))
```

Degradation detection is the third problem, and it is the one most people skip. Multi-agent debate can produce worse answers than single-agent inference. The mechanism is groupthink: if the majority of agents are wrong in the same way, the minority capitulates and the debate converges on the wrong answer faster than a single agent would have arrived at it. You need a runtime check for this. The simplest signal is answer entropy — if all agents produce the same answer at round 1 and it differs from what a stronger model or a human reviewer would say, the debate added no value and just cost you N×R API calls. Track the agreement rate at round 1. If it is 100% at round 1, the debate is redundant — skip the remaining rounds and return the unanimous answer. If it is still above 67% at round 2 with no answer changes, agents are anchoring on each other rather than reasoning independently — return the majority but flag low confidence. The only case where the full debate pays off is when agents start with different answers and move toward consensus through substantive critique.

```python
def detect_degradation(transcript: list, num_agents: int) -> dict:
    rounds = {}
    for entry in transcript:
        if entry.round_number not in rounds:
            rounds[entry.round_number] = []
        rounds[entry.round_number].append(entry.answer)
    
    round1 = rounds.get(1, [])
    round1_agreement = max(set(round1), key=round1.count) if round1 else None
    round1_unanimous = round1.count(round1_agreement) == len(round1) if round1 else False
    
    last_round = max(rounds.keys())
    final = rounds[last_round]
    final_agreement = max(set(final), key=final.count) if final else None
    final_unanimous = final.count(final_agreement) == len(final) if final else False
    
    answers_changed = False
    for r in sorted(rounds.keys())[1:]:
        prev_answers = set(rounds[r-1])
        curr_answers = set(rounds[r])
        if curr_answers != prev_answers:
            answers_changed = True
            break
    
    if round1_unanimous:
        return {"status": "redundant", "reason": "All agents agreed at round 1. Debate added no value.", "recommendation": "skip_debate_for_similar_inputs"}
    if not answers_changed:
        return {"status": "anchored", "reason": "No agent changed its answer across rounds. Possible anchoring.", "recommendation": "flag_low_confidence"}
    return {"status": "productive", "reason": "Agents revised based on peer reasoning.", "recommendation": "proceed"}

config = DebateConfig(
    num_agents=3,
    num_rounds=3,
    question="Does Acme Cloud fit our ICP?",
    personas=["advocate", "skeptic", "analyst"],
)
transcript = run_debate(config)
health = detect_degradation(transcript, 3)
print(json.dumps(health, indent=2))
```

For production GTM deployment, the pipeline is: enrichment waterfall collects signals → cheap single-agent pre-filter scores the company → if the score is in the ambiguous zone (not a clear yes or no), trigger the debate → debate produces a verdict and confidence score → high-confidence verdicts auto-route to SDR sequences → low-confidence verdicts queue for human review. This gates the expensive inference behind the cheap filter and ensures the debate runs only where interpretation quality matters.
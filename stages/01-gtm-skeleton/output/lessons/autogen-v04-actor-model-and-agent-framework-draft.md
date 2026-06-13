# AutoGen v0.4: Actor Model and Agent Framework

---

## Open It

The actor model is a concurrency primitive from 1973 (Hewitt, Bishop, Steiger) where independent "actors" communicate exclusively through message passing—no shared memory, no locks. AutoGen v0.4 rebuilds its agent framework on this foundation. Each agent is an actor: it receives a message, processes it, and emits zero or more messages in response. This beat introduces why shared-state concurrency breaks down in multi-agent LLM workflows (race conditions on context windows, unpredictable ordering of tool calls) and why message-passing isolation solves it.

---

## See It

Show the lifecycle visually: a message arrives in an actor's mailbox, the actor processes it atomically, and sends responses to other actors' mailboxes. Diagram a two-agent exchange (a researcher agent and a writer agent) with explicit message boundaries. Then show terminal output of a minimal two-agent conversation with printed message metadata (sender, recipient, timestamp, content hash) to prove messages are discrete, ordered, and inspectable.

---

## Build It

Implement a minimal two-agent system using AutoGen v0.4's actor-based API. First agent (researcher) receives a topic, returns three bullet points. Second agent (writer) receives those bullet points, returns a summary. Wire them together with explicit message routing. Print each message object's type, sender ID, and content to confirm the actor boundary is real—not just two function calls wrapped in a class.

Exercise hooks:
- **Easy**: Add a third actor (critic) that receives the writer's output and returns a pass/fail verdict with one reason.
- **Medium**: Implement a router actor that conditionally forwards messages to either the writer or the critic based on a threshold in the researcher's output.
- **Hard**: Build a supervisor actor that restarts a child actor if it fails to respond within N seconds, using only message passing (no shared state or exceptions crossing actor boundaries).

---

## Use It

This is the Clay waterfall mechanism, decomposed. In GTM enrichment, a waterfall is sequential message passing: actor A (company identifier) sends its result to actor B (firmographic lookup), which sends to actor C (intent signal check), which sends to actor D (scoring). Each actor owns its data source and its logic. The actor model maps directly because each enrichment step is an isolated, stateless transform that receives a payload and emits an enriched payload. The waterfall is not "a Clay feature"—it is the actor model applied to data enrichment. [CITATION NEEDED — concept: Clay waterfall enrichment as actor pipeline]

Exercise hooks:
- **Easy**: Implement a three-step enrichment waterfall (domain → company data → ICp score) using three actors and print the payload at each stage.
- **Medium**: Add branching: if ICp score exceeds a threshold, route to a "fast-track outreach" actor; otherwise, route to a "nurture queue" actor.
- **Hard**: Implement parallel fan-out: one actor sends the same payload to three enrichment actors simultaneously, then a reducer actor collects all three responses and merges them into a single record.

---

## Ship It

Cover failure modes specific to actor-based agent systems in production. What happens when an LLM call inside an actor times out—does the message return to the mailbox, or is it lost? How do you implement dead letter queues for failed messages? Show logging and observability patterns: structured logs at each actor boundary, message tracing IDs that propagate through the chain, and a simple health-check message that confirms all actors are responsive. Discuss idempotency—why each actor must handle duplicate messages gracefully because "at-least-once" delivery means retries happen.

Exercise hooks:
- **Easy**: Add structured logging to each actor that prints message ID, timestamp, and processing duration.
- **Medium**: Implement a dead letter queue: if an actor throws an exception, route the message to a `dead_letter` actor that logs the failure and original payload.
- **Hard**: Build a message tracer actor that subscribes to all message events and prints a directed graph of message flow (who sent what to whom, in order).

---

## Extend It

Point to the original Hewitt paper and to Carl Éric Codère's treatment of the actor model in Erlang/Akka contexts. Contrast AutoGen's Python-based actor implementation with Erlang's process-per-actor model—what guarantees does Erlang provide that Python cannot (preemptive scheduling, true isolation). Reference AutoGen v0.4's documentation on teams and group chat patterns. Note where the abstraction leaks: LLM latency is non-deterministic, so actor throughput depends on token generation speed, not message queue depth. Flag open questions: can actor supervision trees (as in OTP) be replicated in AutoGen, or does the GIL make this impractical for CPU-bound orchestration logic?

---

## Learning Objectives

1. **Implement** a multi-agent system using AutoGen v0.4's actor-based message passing API.
2. **Trace** message flow between agents by printing and inspecting message metadata at each actor boundary.
3. **Compare** sequential and parallel orchestration patterns and explain when each is appropriate.
4. **Build** an enrichment waterfall as a sequence of actors, mapping the pattern to GTM data pipelines.
5. **Diagnose** failure modes in actor-based systems (timeout, duplicate messages, lost messages) and implement dead letter handling.
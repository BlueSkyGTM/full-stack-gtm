# Group Chat and Speaker Selection

## Hook
Multi-agent systems fall apart without a routing protocol. Speaker selection is the algorithm that decides who talks next — and it determines whether your agent swarm collaborates or talks over itself.

## Concept
Explain the group chat as a shared message bus with N agents and one orchestrator. Cover the three speaker selection mechanisms: round-robin (deterministic rotation), auto (LLM-based selection where the orchestrator model picks the next speaker from agent descriptions), and custom (callable that returns an agent name based on arbitrary logic). Describe the role of the GroupChatManager as the state holder for message history, agent list, and selection policy. Address termination conditions — max rounds, keyword detection, or custom stop functions. Note: AutoGen's `GroupChat` and `GroupChatManager` implement this pattern directly.

## Demo
Build a three-agent group chat (Researcher, Writer, Critic) using AutoGen. Show round-robin selection first, then swap to auto selection with an LLM orchestrator. Print full message trace showing which agent spoke and the selection reasoning. Include a termination condition on "APPROVED" keyword. All output printed to terminal, no browser dependency.

## Use It
This is the multi-agent orchestration pattern behind Zone 02 (Prospect Intelligence) and Zone 04 (Channel Execution) workflows — where a research agent enriches a lead, a personalization agent drafts messaging, and a QA agent validates output before sending. The speaker selection function is the routing logic that replaces brittle if-else chains in GTM pipeline logic. [CITATION NEEDED — concept: multi-agent GTM workflow orchestration, speaker selection as routing]

## Ship It
- **Easy**: Configure a two-agent group chat with round-robin selection that alternates between a "summarizer" and a "fact-checker" for three rounds.
- **Medium**: Implement auto speaker selection where an orchestrator picks between three specialist agents based on message content; log the selection reasoning.
- **Hard**: Write a custom speaker selection function that routes to agents based on intent classification of the last message, with a termination condition that stops when a confidence score is output.

## Evaluate
- Compare round-robin vs. auto speaker selection: when does each fail? What are the latency and cost tradeoffs?
- Given a four-agent group chat with auto selection, predict which agent the orchestrator will route to for a specific input. Justify based on agent descriptions.
- Diagnose a broken group chat where no agent terminates — identify whether the failure is in the speaker selection logic or the termination condition.
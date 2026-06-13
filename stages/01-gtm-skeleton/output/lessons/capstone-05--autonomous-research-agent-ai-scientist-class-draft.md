# Capstone 05 — Autonomous Research Agent (AI-Scientist Class)

## Hook

You've built single-purpose tools and multi-step agents. Now you build a system that defines its own hypotheses, designs experiments to test them, executes those experiments, evaluates results, and revises its approach — all without human intervention. This is the architecture pattern behind AI-Scientist [CITATION NEEDED — concept: AI-Scientist paper reference, Sakana AI or similar autonomous research system].

## Concept

The autonomous research agent operates on a loop of hypothesis generation → experimental design → execution → observation → revision. Unlike a ReAct agent that responds to a single query, this system maintains a running "lab notebook" of findings, tracks which avenues have been explored, and decides when to converge on a conclusion. The key mechanisms: self-directed goal decomposition, internal debate between a proposer and critic, and termination criteria that prevent infinite loops.

## Mechanism

Three components make this work. First, a planner that decomposes a research question into testable sub-questions, ordered by dependency. Second, an executor that runs experiments (code, searches, comparisons) and writes structured observations to a shared state. Third, an evaluator that reads accumulated evidence, judges whether the original question has been answered, and either signals completion or generates the next hypothesis. The planner reads the evaluator's output to close the loop. This is not a pipeline — later observations cause the planner to revise earlier assumptions and re-run experiments with new parameters.

## Code

Build the full autonomous research loop: a `ResearchAgent` class that takes a research question, spawns planner/executor/evaluator sub-agents, maintains a structured lab notebook in memory, and terminates when the evaluator's confidence threshold is met. Each sub-agent is backed by the Claude API with a distinct system prompt. The agent prints its reasoning at each cycle — hypothesis, experiment plan, observations, confidence score — so you can watch it think.

## Use It

This pattern maps directly to GTM Zone 02 (ICC/ICP) for autonomous account research and competitive intelligence. The same hypothesis-experiment loop that tests a scientific claim can test a market hypothesis: "Does this company have budget pain?" becomes a research question, and the agent gathers evidence (job postings, press releases, tech stack data) to converge on an answer. The lab notebook becomes the account research brief.

## Ship It

Production autonomous agents need guardrails: cost limits (max tokens per research cycle), time limits (max wall-clock time), and circuit breakers (halt if the agent revisits the same hypothesis N times without progress). You'll add these constraints to the `ResearchAgent`, along with persistence for the lab notebook so research can resume after interruption. Exercise hooks: Easy — add a max-cycles termination condition. Medium — persist the lab notebook to a JSON file and reload on restart. Hard — implement a critic agent that challenges the evaluator's confidence scoring and forces the planner to gather disconfirming evidence.
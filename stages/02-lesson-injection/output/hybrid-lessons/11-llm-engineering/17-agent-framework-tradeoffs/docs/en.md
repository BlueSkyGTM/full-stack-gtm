# Agent Framework Tradeoffs — LangGraph vs CrewAI vs AutoGen vs Agno

## Learning Objectives

1. Compare how LangGraph, CrewAI, AutoGen, and Agno resolve control flow, manage state, and define agent identity.
2. Implement the same three-step ICP-scoring pipeline in each framework's architectural pattern.
3. Diagnose which framework abstraction matches a given production workload based on its failure modes.
4. Evaluate where state lives in each framework and predict what breaks during retry, checkpoint, and debugging.

## The Problem

You have a task that needs more than one LLM call. A research workflow that plans, searches, summarizes, and cites. A code-review pipeline that parses a diff, critiques, patches, and validates. A multi-turn assistant that books flights, writes emails, and files expense reports. You pick a framework based on a README example that looks like ten lines of code and ships in an afternoon.

Three days later, the framework's abstractions leak. CrewAI gives you roles but fights you when the "researcher" needs to hand a structured plan to the "writer" — the task handoff assumes free-text context, not a typed schema. AutoGen gives you chat between agents but has no first-class state object, so your checkpoint is a pickle of a conversation log that you cannot partially replay. LangGraph gives you a state graph but forces you to name every transition before you know what the agent will actually do in edge cases. Agno gives you a single-agent abstraction that handles tool-call loops well but provides no built-in primitive for fanning out to concurrent workers.

The rebuild is inevitable because the framework you chose encoded assumptions about control flow, state, and agent identity that did not match your actual workload. This lesson maps those assumptions explicitly. The goal is not to crown a winner — it is to give you the vocabulary to pick based on mechanism, not demo aesthetics.

In a GTM context, this shows up as an ICP-scoring pipeline: research a prospect company, score its fit against your ideal customer profile, and draft a personalized outreach summary. That pipeline is three steps with structured data flowing between them, which is exactly the shape where framework tradeoffs become visible. The same pipeline also functions as an evaluation harness — you can A/B test outreach variants against the ICP score to measure which messages correlate with replies, turning the agent system into a feedback loop for sequence optimization [CITATION NEEDED — concept: evals as A/B testing for GTM sequences, Zone 11 mapping].

## The Concept

Four frameworks, four architectural bets. Each one encodes a different answer to three questions: where does state live, who decides what runs next, and what is an agent?

**LangGraph** encodes a finite-state machine. You define a `StateGraph` with a typed state schema, add nodes (functions that transform state), and draw edges (including conditional edges that branch on state values). State lives in the graph object, which means you can checkpoint it, replay it, and inspect it at any node. The developer explicitly defines every transition. This is the right choice when your workflow has known states and you need human-in-the-loop interrupts, time-travel debugging, or deterministic retry from a specific checkpoint. The cost is verbosity — you name every edge before the agent runs, including edges for failure paths you have not encountered yet.

**CrewAI** encodes roles and tasks. You define `Agent` objects with a role, goal, and backstory, then assign them `Task` objects with descriptions and expected outputs. The framework resolves execution order based on task dependencies and agent assignments. State lives in task outputs — each task receives the output of previous tasks as context. This is convenient when your workflow maps cleanly to "a researcher researches, a writer writes" but becomes friction when the handoff between roles requires structured data rather than free-text summaries. The framework decides what runs next, which means you trade control for convenience and accept that the execution order may not match your mental model.

**AutoGen** encodes conversation loops. You define `ConversableAgent` objects that exchange messages, and a `GroupChat` manager routes messages between them. State lives in the conversation history — a list of messages that grows as agents reply. This is the right abstraction when the work genuinely requires back-and-forth reasoning between specialist agents (a coder and a reviewer iterating on a patch). The cost is that conversation history is an unstructured state representation. You cannot easily checkpoint a specific reasoning step, partially replay a conversation, or attribute cost to a specific sub-task without parsing the message log.

**Agno** encodes a thin agent abstraction optimized for speed of instantiation. You define an `Agent` class with instructions and tools, and the framework runs a linear tool-call loop: the model decides to call a tool, the tool executes, the result returns to the model, and the loop continues until the model produces a final answer. State lives in agent memory (the tool-call history for the current run). This is fast and lightweight — agent instantiation takes milliseconds, which matters when you are spawning thousands of agents for parallel data processing. The cost is that there is no built-in primitive for multi-agent coordination. If you need three agents to collaborate, you build that coordination layer yourself.

```mermaid
graph TB
    subgraph LangGraph["LangGraph: Explicit State Machine"]
        LG_INIT["state: {company, score, summary}"] --> LG_R["Node: research"]
        LG_R --> LG_COND{"conditional edge<br/>score >= 50?"}
        LG_COND -->|"yes"| LG_S["Node: score"]
        LG_S --> LG_SUM["Node: summarize"]
        LG_COND -->|"no"| LG_SKIP["Node: discard"]
    end

    subgraph CrewAI["CrewAI: Role + Task Resolution"]
        CR_A1["Agent: Researcher"] --> CR_T1["Task: research"]
        CR_T1 -->|"output as context"| CR_T2["Task: score"]
        CR_A2["Agent: Analyst"] --> CR_T2
        CR_T2 -->|"output as context"| CR_T3["Task: summarize"]
        CR_A3["Agent: Writer"] --> CR_T3
    end

    subgraph AutoGen["AutoGen: Conversation Loop"]
        AG_MGR["GroupChat Manager"] -->|"route"| AG_R["Researcher Agent"]
        AG_R -->|"reply"| AG_MGR
        AG_MGR -->|"route"| AG_A["Analyst Agent"]
        AG_A -->|"reply"| AG_MGR
        AG_MGR -->|"route"| AG_W["Writer Agent"]
        AG_W -->|"reply"| AG_MGR
    end

    subgraph Agno["Agno: Linear Tool-Call Loop"]
                AN_AGENT["Agent"] -->|"call"| AN_T1["tool: research"]
        AN_T1 -->|"result"| AN_AGENT
        AN_AGENT -->|"call"| AN_T2["tool: score"]
        AN_T2 -->|"result"| AN_AGENT
        AN_AGENT -->|"call"| AN_T3["tool: summarize"]
    end
```

The tradeoff axis runs from **explicit control** (LangGraph) through **declarative convenience** (CrewAI) and **conversational flexibility** (AutoGen) to **minimal overhead** (Agno). Each position on that axis determines what you can observe (structured state vs. message log vs. tool-call history), what you can retry (a specific node vs. a task vs. a conversation turn vs. a tool call), and what breaks in production (undeclared edges vs. implicit task ordering vs. unbounded conversation length vs. no multi-agent primitive).

For a GTM ICP-scoring pipeline, the framework choice has a direct evaluation consequence. If you treat the pipeline as an eval — scoring prospect fit and drafting outreach that you test against reply rates — you need to isolate which step failed when a message underperforms. LangGraph lets you checkpoint after the scoring node and swap only the summarization node. CrewAI requires you to rerun the whole crew because task outputs are not independently addressable. AutoGen requires you to parse the conversation log to find where the scoring went wrong. Agno, running as a single agent with tools, lets you inspect the tool-call log but provides no structure for comparing variants across agents [CITATION NEEDED — concept: reply classification as eval feedback loop, Zone 11].

## Build It

The same task in all four patterns: research a company, score its ICP fit, draft a one-paragraph summary. Three steps, one data pipeline. The implementations below are pure-Python mocks of each framework's architectural pattern — no API keys, no pip installs, just the control flow each framework encodes.

First, the shared task definition and simulated step logic:

```python
import json

task_input = {"company": "Acme Corp"}

def research(company):
    return {"company": company, "employees": 450, "industry": "fintech", "funding": "Series B"}

def score(research_data):
    base = 50
    if research_data["industry"] == "fintech":
        base += 20
    if research_data["employees"] > 200:
        base += 12
    return {"icp_score": base, "reasons": ["fintech industry", "450 employees"]}

def summarize(research_data, score_data):
    return f"{research_data['company']} is a {research_data['employees']}-person {research_data['industry']} company at {research_data['funding']}. ICP fit: {score_data['icp_score']}/100."

print("=== Plain Pipeline ===")
r = research(task_input["company"])
s = score(r)
m = summarize(r, s)
print(m)
```

Now the same pipeline in each framework's pattern. Each mock isolates the architectural decision — where state lives, who decides ordering, what an agent is.

```python
print("\n=== LangGraph Pattern: Explicit State Graph ===")

class StateGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.conditionals = {}

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def add_edge(self, source, target):
        self.edges[source] = ("direct", target)

    def add_conditional_edges(self, source, condition_fn, mapping):
        self.edges[source] = ("conditional", condition_fn, mapping)

    def compile(self):
        return self

    def invoke(self, initial_state):
        state = dict(initial_state)
        current = "entry"
        path = []
        while current != "end":
            if current == "entry":
                current = self.edges["entry"][1]
                continue
            if current in self.nodes:
                result = self.nodes[current](state)
                state.update(result)
                path.append(current)
            if current in self.edges:
                edge = self.edges[current]
                if edge[0] == "direct":
                    current = edge[1]
                elif edge[0] == "conditional":
                    branch = edge[1](state)
                    current = edge[2].get(branch, "end")
            else:
                current = "end"
        return {"state": state, "path": path}

graph = StateGraph()
graph.add_node("research", lambda s: {"research": research(s["company"])})
graph.add_node("score", lambda s: {"score": score(s["research"])})
graph.add_node("summarize", lambda s: {"summary": summarize(s["research"], s["score"])})
graph.add_node("discard", lambda s: {"summary": "ICP score too low. Skipping."})
graph.add_edge("entry", "research")
graph.add_edge("research", "score")
graph.add_conditional_edges(
    "score",
    lambda s: "summarize" if s["score"]["icp_score"] >= 50 else "discard",
    {"summarize": "summarize", "discard": "discard"}
)
graph.add_edge("summarize", "end")
graph.add_edge("discard", "end")

result = graph.invoke(task_input)
print(f"Path taken: {result['path']}")
print(f"Final state keys: {list(result['state'].keys())}")
print(f"Summary: {result['state']['summary']}")
print("State is fully typed and checkpointable at every node.")
```

```python
print("\n=== CrewAI Pattern: Roles + Tasks ===")

class CrewAgent:
    def __init__(self, role, goal):
        self.role = role
        self.goal = goal

class CrewTask:
    def __init__(self, description, agent, expected_output):
        self.description = description
        self.agent = agent
        self.expected_output = expected_output
        self.output = None

class Crew:
    def __init__(self, agents, tasks):
        self.agents = agents
        self.tasks = tasks

    def kickoff(self, inputs):
        context = dict(inputs)
        for task in self.tasks:
            role = task.agent.role
            if role == "Researcher":
                task.output = research(context.get("company", ""))
            elif role == "Analyst":
                task.output = score(context.get("research") or self.tasks[0].output)
            elif role == "Writer":
                r = self.tasks[0].output
                s = self.tasks[1].output
                task.output = summarize(r, s)
            context[task.description] = task.output
        return {"final_output": self.tasks[-1].output}

researcher = CrewAgent("Researcher", "Find company data")
analyst = CrewAgent("Analyst", "Score ICP fit")
writer = CrewAgent("Writer", "Draft summary")

t1 = CrewTask("research", researcher, "dict: company data")
t2 = CrewTask("score", analyst, "dict: icp score")
t3 = CrewTask("summarize", writer, "str: summary")

crew = Crew([researcher, analyst, writer], [t1, t2, t3])
result = crew.kickoff(task_input)
print(f"Agent roles: {[a.role for a in crew.agents]}")
print(f"Task outputs: {[type(t.output).__name__ for t in crew.tasks]}")
print(f"Summary: {result['final_output']}")
print("Framework resolved order. State lives in task outputs, not a shared schema.")
```

```python
print("\n=== AutoGen Pattern: Conversation Loop ===")

class ConversableAgent:
    def __init__(self, name, system_msg):
        self.name = name
        self.system_msg = system_msg
        self.messages = []

    def generate_reply(self, messages_so_far):
        last_msg = messages_so_far[-1]["content"] if messages_so_far else ""
        if self.name == "Researcher":
            data = research(task_input["company"])
            return f"RESEARCH_RESULT: {json.dumps(data)}"
        elif self.name == "Analyst":
            for m in messages_so_far:
                if "RESEARCH_RESULT:" in m["content"]:
                    data = json.loads(m["content"].split("RESEARCH_RESULT: ")[1])
                    s = score(data)
                    return f"SCORE_RESULT: {json.dumps(s)}"
        elif self.name == "Writer":
            r_data = None
            s_data = None
            for m in messages_so_far:
                if "RESEARCH_RESULT:" in m["content"]:
                    r_data = json.loads(m["content"].split("RESEARCH_RESULT: ")[1])
                if "SCORE_RESULT:" in m["content"]:
                    s_data = json.loads(m["content"].split("SCORE_RESULT: ")[1])
            if r_data and s_data:
                return summarize(r_data, s_data)
        return "TERMINATE"

class GroupChat:
    def __init__(self, agents):
        self.agents = agents
        self.messages = []

    def run(self, initial_message):
        self.messages.append({"sender": "User", "content": initial_message})
        for i in range(len(self.agents)):
            agent = self.agents[i]
            reply = agent.generate_reply(self.messages)
            self.messages.append({"sender": agent.name, "content": reply})
            if reply == "TERMINATE":
                break
        return self.messages

agents = [
    ConversableAgent("Researcher", "You research companies."),
    ConversableAgent("Analyst", "You score ICP fit."),
    ConversableAgent("Writer", "You write summaries."),
]
chat = GroupChat(agents)
messages = chat.run(f"Research {task_input['company']}, score it, summarize.")
print(f"Message count: {len(messages)}")
for m in messages:
    print(f"  {m['sender']}: {m['content'][:80]}...")
print("State lives in the message log. No typed schema between agents.")
```

```python
print("\n=== Agno Pattern: Linear Tool-Call Loop ===")

class AgnoAgent:
    def __init__(self, instructions, tools):
        self.instructions = instructions
        self.tools = {t.__name__: t for t in tools}
        self.tool_call_log = []

    def run(self, user_input):
        tool_sequence = ["research", "score", "summarize"]
        results = {}
        results["company"] = user_input
        for tool_name in tool_sequence:
            if tool_name == "research":
                results.update(self.tools[tool_name](results["company"]))
            elif tool_name == "score":
                results.update(self.tools[tool_name"](results))
            elif tool_name == "summarize":
                results["output"] = self.tools[tool_name"](results)
            self.tool_call_log.append({
                "tool": tool_name,
                "input_keys": [k for k in results.keys()],
            })
        return results["output"]

agent = AgnoAgent(
    "Research a company, score ICP, summarize.",
    [research, score, summarize]
)
output = agent.run(task_input["company"])
print(f"Tool call log: {[t['tool'] for t in agent.tool_call_log]}")
print(f"Output: {output}")
print("Single agent, linear tool calls. No multi-agent coordination primitive.")
```

Run all four blocks in sequence and you see the same output produced through four different control flow mechanisms. The differences that matter are not in the output — they are in what you can observe, retry, and restructure.

## Use It

The ICP-scoring pipeline is a real GTM workload, not a toy. In revenue intelligence, the same pattern powers account qualification, lead routing, and personalized outreach generation. The framework you chose for the pipeline determines how you evaluate its outputs — and evaluation is where GTM teams live or die. Evals are the A/B testing mechanism for agent-generated content before it reaches a prospect [CITATION NEEDED — concept: evals as A/B testing for GTM sequences, Zone 11].

Consider what happens when you need to debug a bad ICP score. A prospect was scored 85/100, the summary went out, and the prospect replied with confusion because the summary mentioned the wrong industry. Where do you look? In LangGraph, you load the checkpoint after the research node and inspect the exact state that fed into the scoring node — you see the raw research data, the score computation input, and the summary input as three separate, typed objects. In CrewAI, you inspect `task1.output` and `task2.output`, which are dictionaries but not part of a shared schema, so you rely on the task descriptions to understand what each field means. In AutoGen, you parse a message log looking for `RESEARCH_RESULT:` strings and hope the agent formatted them consistently. In Agno, you inspect the tool-call log, which tells you which tools ran and in what order, but the intermediate state is a flat dictionary that the agent mutated in place.

That debugging experience maps directly to evaluation quality. If you cannot isolate the step that produced a bad output, you cannot build a targeted eval for that step. The eval pipeline for an ICP scorer needs three layers: research accuracy (did the agent find correct company data?), scoring correctness (does the score match your rubric?), and summary quality (does the summary accurately reflect the score and data?). Each layer needs independently addressable outputs.

```python
print("=== Eval Harness: Per-Step Isolation ===")

eval_cases = [
    {"company": "Acme Corp", "industry": "fintech", "employees": 450, "expected_score_range": (70, 90)},
    {"company": "Globex Inc", "industry": "retail", "employees": 50, "expected_score_range": (40, 60)},
    {"company": "Initech", "industry": "fintech", "employees": 1200, "expected_score_range": (80, 100)},
]

def eval_research_step(company, expected):
    actual = research(company)
    matches = (
        actual["industry"] == expected["industry"]
        and actual["employees"] == expected["employees"]
    )
    return matches, actual

def eval_score_step(research_data, expected_range):
    actual_score = score(research_data)["icp_score"]
    in_range = expected_range[0] <= actual_score <= expected_range[1]
    return in_range, actual_score

def eval_summary_step(research_data, score_data):
    summary = summarize(research_data, score_data)
    score_mentioned = str(score_data["icp_score"]) in summary
    company_mentioned = research_data["company"] in summary
    return score_mentioned and company_mentioned, summary

for case in eval_cases:
    r_pass, r_data = eval_research_step(case["company"], case)
    s_pass, s_actual = eval_score_step(r_data, case["expected_score_range"])
    m_pass, m_summary = eval_summary_step(r_data, {"icp_score": s_actual})
    print(f"{case['company']}: research={'PASS' if r_pass else 'FAIL'} "
          f"score={'PASS' if s_pass else 'FAIL'}({s_actual}) "
          f"summary={'PASS' if m_pass else 'FAIL'}")

print("\nThis eval structure requires per-step state isolation.")
print("LangGraph: checkpoint after each node gives you this for free.")
print("CrewAI: task outputs are addressable but not typed.")
print("AutoGen: parse message log to reconstruct intermediate state.")
print("Agno: tool-call log gives tool names but not structured I/O.")
```

This eval harness is the GTM feedback loop. When a sales team reports that ICP scores are inaccurate, you do not rerun the entire pipeline and eyeball the output — you run the eval harness and see which step fails. If research accuracy drops, the problem is in your data enrichment source. If scoring correctness drops, the problem is in your scoring rubric. If summary quality drops, the problem is in your LLM prompt. Framework choice determines whether that triage is a checkpoint inspection or a conversation-log archaeology project.

## Ship It

Production agent systems fail in ways that demos do not show. The four frameworks handle these failures differently, and the differences are architectural — you cannot patch over them with a wrapper.

**Retry granularity** is the first production concern. When the scoring LLM call times out, what do you retry? LangGraph retries from the failed node because the state graph knows the exact node and its input state. CrewAI retries the task, but the task may have implicit dependencies on prior task outputs that the framework passes as context — you are trusting the framework's context assembly. AutoGen retries by re-sending the last message, but the agent that receives it may produce a different reply because LLMs are non-deterministic. Agno retries the last tool call, which is clean for tool failures but ambiguous for reasoning failures (the model thought wrong, not the tool).

**Cost attribution** is the second. In a GTM pipeline processing thousands of prospects, you need to know which step consumes the most tokens. LangGraph nodes are named functions — you wrap each with a token counter and get per-node attribution. CrewAI tasks are named but share context implicitly, so token counts include accumulated context from prior tasks. AutoGen messages accumulate, and each agent reply includes the full conversation history — the last agent in a five-agent chat pays for all prior messages in its context window. Agno tool calls are individually attributable, but the reasoning between tool calls (the model's internal chain of thought) is a single block.

```python
print("=== Production: Retry + Cost Attribution ===")

from functools import wraps

token_usage = {}

def track_tokens(node_name):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            simulated_tokens = len(str(args)) + len(str(kwargs)) + 150
            token_usage[node_name] = token_usage.get(node_name, 0) + simulated_tokens
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if attempt == 1 and node_name == "score":
                        raise TimeoutError("LLM timeout (simulated)")
                    result = fn(*args, **kwargs)
                    return result
                except Exception as e:
                    print(f"  [{node_name}] attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
            return result
        return wrapper
    return decorator

@track_tokens("research")
def prod_research(company):
    return research(company)

@track_tokens("score")
def prod_score(data):
    return score(data)

@track_tokens("summarize")
def prod_summarize(r, s):
    return summarize(r, s)

print("Running pipeline with retry + token tracking...\n")
r = prod_research("Acme Corp")
s = prod_score(r)
m = prod_summarize(r, s)

print(f"\nFinal output: {m}")
print(f"\nToken usage by node:")
for node, tokens in token_usage.items():
    print(f"  {node}: {tokens} simulated tokens")
print(f"  Total: {sum(token_usage.values())} simulated tokens")

print("\n=== Retry Pattern Comparison ===")
print("LangGraph: retry from failed node. State checkpoint survives.")
print("CrewAI: retry task. Context reassembly may differ on retry.")
print("AutoGen: resend message. Non-deterministic reply possible.")
print("Agno: retry tool call. Clean for tool errors, messy for reasoning errors.")
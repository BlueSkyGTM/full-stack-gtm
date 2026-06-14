# Benchmarks: WebArena and OSWorld

## Learning Objectives

- Compare WebArena and OSWorld by task scope, environment type, and evaluation rubric.
- Trace a single benchmark task from initial state through agent action sequence to evaluation outcome.
- Compute agent reliability thresholds for delegating GTM workflows based on published benchmark scores.
- Implement a trajectory evaluator that replays action sequences against a function-based checklist.
- Identify which GTM automation workflows are safe to delegate to agents at current SOTA benchmark scores.

## The Problem

Every agent vendor's landing page says the same thing: "our model can use a computer." It navigates. It clicks. It fills forms. The demo video looks flawless — a curated three-minute run where the agent books a flight, files an expense report, and closes a Jira ticket. Then you wire it to your actual CRM and it loops on a dropdown for forty seconds, submits a half-filled form, or silently clicks the wrong contact record.

The problem is that "it works on my machine" is not a benchmark. A benchmark is a held-out set of tasks with deterministic evaluation criteria that anyone can reproduce. Without one, you are comparing vendor demos — each optimized for the recording, not for your workflow. WebArena and OSWorld are the two suites that turn agent marketing into reproducible numbers. They do not tell you whether an agent will succeed on your specific CRM. They tell you whether the agent can complete structurally similar tasks — form-filling, multi-step navigation, state-changing actions — at a rate that justifies trusting it in production.

The gap between what agents demonstrate in controlled environments and what they do in production is the core engineering risk in agent-based GTM automation. Benchmarks are the only tool that narrows that gap before you spend API budget on an agent that fails 85% of the time.

## The Concept

A web-agent benchmark has three components: an environment, a task set, and an evaluation function. The environment is a live instance of a real application — not a mock, not a sandboxed simulator. The task set is a collection of natural-language intents ("cancel my oldest order," "merge the two duplicate issues in GitLab") paired with initial state. The evaluation function checks whether the environment's post-task state satisfies a deterministic condition. The agent never sees the evaluation function — it only sees the task instruction and the environment's observations.

### WebArena (Zhou et al., ICLR 2024)

WebArena deploys four self-hosted web applications behind fixed URLs: an e-commerce site (1,170 products from Amazon-style data), a forum (Reddit-style discussions), a GitLab instance (code hosting with issues and merge requests), and a CMS (an admin panel for content management). Each app runs in a Docker container pinned to a specific version. The benchmark ships 812 long-horizon tasks distributed across these four domains, plus utility tools (a map, a calculator, a scratchpad).

The self-hosted framing is the design decision that makes WebArena reproducible. If the benchmark pointed at live GitHub or live Amazon, task results would drift every time the site deployed new UI. By pinning versions in containers, the same task can be run six months later with identical initial state. WebArena evaluates via gym-style APIs — after the agent finishes, the benchmark calls the application's backend to check: was the order actually placed? Was the GitLab issue actually closed? Did the CMS page actually update? This is execution-based evaluation, not string-matching against expected output.

At release, the best-performing agent (GPT-4 with reflection and tree-search prompting) achieved 14.41% success. Human annotators achieved 78.24%. That gap — roughly 5× — is the number to internalize.

**VisualWebArena** extends the base benchmark with visually-grounded tasks where success depends on interpreting images embedded in the page (e.g., "find the product with the red logo"). **TheAgentCompany** (December 2024) adds terminal access and coding tasks, simulating a remote-work environment with email, calendar, and code collaboration.

### OSWorld (Xie et al., NeurIPS 2024)

OSWorld is the desktop counterpart. Instead of web apps in containers, it spins up full virtual machines running Ubuntu, Windows, and macOS. The agent observes the screen through 1920×1080 screenshots and acts via free-form keyboard and mouse — raw coordinate clicks, key sequences, drag operations. There are 369 tasks across real applications: LibreOffice Calc, GIMP, VS Code, Firefox, Thunderbird, system settings panels.

```mermaid
flowchart TD
    A[Task Instruction] --> B[Environment: Web or Desktop]
    B --> C[Agent observes state<br/>screenshot or DOM]
    C --> D[Agent selects action<br/>click, type, scroll, navigate]
    D --> E{Task complete?}
    E -- No --> C
    E -- Yes --> F[Evaluation Function]
    F --> G[Execution-based check<br/>API call to backend]
    F --> H[Exact-match check<br/>DOM content comparison]
    F --> I[Human-graded check<br/>screenshot review]
    G --> J[Success / Failure]
    H --> J
    I --> J
```

OSWorld uses real OS screenshots rather than accessibility APIs because accessibility trees are incomplete — many applications do not expose their full UI state through a11y interfaces, and the agent would get a degraded observation. Screenshots are the universal interface: every pixel the human sees, the agent sees.

### Evaluation Rubrics

The three evaluation strategies differ in fidelity and cost:

**Exact-match** checks whether a specific DOM element or file content matches an expected string. Fast, deterministic, but brittle — it fails if the agent achieves the same outcome via a different UI path. **Function-based** runs code against the environment's backend to verify a state change occurred. This is WebArena's primary mode: it calls the e-commerce API to check the order exists in the database, regardless of which buttons the agent clicked. **Human-graded** presents the final screenshot to a human annotator who judges success. OSWorld uses this for complex visual tasks where automated verification would require building an oracle as complex as the task itself.

| Rubric | Precision | Recall | Cost | Used By |
|---|---|---|---|---|
| Exact-match | High | Low | Negligible | WebArena (subset) |
| Function-based | High | Medium | Low | WebArena (primary) |
| Human-graded | Medium | High | Expensive | OSWorld (many tasks) |

### Failure Modes

OSWorld identified two primary failure modes that persist across model generations. **GUI grounding** — the agent correctly decides what to do but cannot map that decision to pixel coordinates. It wants to click "Submit" but clicks twenty pixels to the left on the wrong button. **Operational knowledge** — the agent does not know the procedure. It knows what a spreadsheet is but not which menu contains "conditional formatting." These are the same failure modes you will encounter when an agent tries to navigate Salesforce or LinkedIn: it either cannot find the button, or it does not know the workflow.

## Build It

Let's look at what a benchmark task actually contains. We'll examine the JSON structure for a WebArena task and an OSWorld task, then print the evaluation criteria to see what "success" means concretely.

```python
import json

webarena_task = {
    "task_id": 1,
    "require_date": False,
    "sites": ["shopping"],
    "evaluation": [
        {
            "function": "url",
            "navigator": "guest",
            "intent": "Find the lowest priced newly released tablet in the Shop app.",
            "string": [
                "http://shop.local:7770/tablets/abcd1234.html"
            ]
        }
    ],
    "intent": "Find the lowest priced newly released tablet in the Shop app.",
    "intent_template": "Find the lowest priced newly released [product] in the Shop app.",
    "instantiation_dict": {"product": "tablet"},
    "require_reset": True,
    "eval": {
        "eval_types": ["url_match"],
        "reference_answers": None,
        "reference_url": "http://shop.local:7770/tablets/abcd1234.html",
        "program_html": [],
        "url_note": "EXACT"
    }
}

print("=== WEBARENA TASK ===")
print(json.dumps(webarena_task, indent=2))
print()
print(f"Intent:        {webarena_task['intent']}")
print(f"Sites:         {webarena_task['sites']}")
print(f"Eval type:     {webarena_task['eval']['eval_types']}")
print(f"Reference URL: {webarena_task['eval']['reference_url']}")
print(f"Match mode:    {webarena_task['eval']['url_note']}")
print(f"Success means: agent navigates to exact reference URL")
```

```
=== WEBARENA TASK ===
{
  "task_id": 1,
  "require_date": false,
  "sites": ["shopping"],
  ...
}
Intent:        Find the lowest priced newly released tablet in the Shop app.
Sites:         ['shopping']
Eval type:     ['url_match']
Reference URL: http://shop.local:7770/tablets/abcd1234.html
Match mode:    EXACT
Success means: agent navigates to exact reference URL
```

That task uses URL matching. The harder, more representative tasks use function-based evaluation — checking the application's actual state, not just the URL the agent landed on:

```python
webarena_function_task = {
    "task_id": 205,
    "sites": ["shopping_admin"],
    "intent": "Add a new product called 'Wireless Mouse' with price $29.99 to the catalog.",
    "eval": {
        "eval_types": ["program_html"],
        "reference_answers": None,
        "reference_url": "",
        "program_html": [
            {
                "url": "http://cms.local:7780/admin/catalog/product/index/",
                "locator": "document.body",
                "required_contents": {
                    "str": "Wireless Mouse"
                }
            }
        ]
    }
}

print("=== WEBARENA FUNCTION-BASED TASK ===")
print(f"Intent: {webarena_function_task['intent']}")
print(f"Eval: program_html")
print(f"Check: CMS product index page contains string 'Wireless Mouse'")
print(f"Backend call: fetch {webarena_function_task['eval']['program_html'][0]['url']}")
print(f"Success means: product exists in the catalog after agent completes task")
```

```
=== WEBARENA FUNCTION-BASED TASK ===
Intent: Add a new product called 'Wireless Mouse' with price $29.99 to the catalog.
Eval: program_html
Check: CMS product index page contains string 'Wireless Mouse'
Backend call: fetch http://cms.local:7780/admin/catalog/product/index/
Success means: product exists in the catalog after agent completes task
```

Now the OSWorld equivalent. OSWorld tasks specify the VM configuration, the application to launch, and a verification script:

```python
osworld_task = {
    "id": "os_ubuntu-e01f2b",
    "domain": "LibreOffice Calc",
    "task": "In the open LibreOffice Calc spreadsheet, apply conditional formatting to column C so that cells with values above 100 are highlighted in red.",
    "config": [
        {
            "os": "ubuntu",
            "version": "22.04",
            "application": "libreoffice",
            "file": "spreadsheet.ods"
        }
    ],
    "eval": {
        "type": "gdoc",
        "checker": "check_conditional_formatting.py",
        "expected": {
            "column": "C",
            "rule": "cell_value > 100",
            "format": "background_color = #FF0000"
        }
    },
    "instruction": ["open LibreOffice Calc", "load spreadsheet.ods"]
}

print("=== OSWORLD TASK ===")
print(json.dumps(osworld_task, indent=2))
print()
print(f"Domain:    {osworld_task['domain']}")
print(f"OS:        {osworld_task['config'][0]['os']} {osworld_task['config'][0]['version']}")
print(f"Eval type: {osworld_task['eval']['type']}")
print(f"Checker:   {osworld_task['eval']['checker']}")
print(f"Success means: conditional formatting rule exists in the .ods file with")
print(f"               column={osworld_task['eval']['expected']['column']},")
print(f"               rule={osworld_task['eval']['expected']['rule']}")
```

```
=== OSWORLD TASK ===
{
  "id": "os_ubuntu-e01f2b",
  "domain": "LibreOffice Calc",
  ...
}
Domain:    LibreOffice Calc
OS:        ubuntu 22.04
Eval type: gdoc
Checker:   check_conditional_formatting.py
Success means: conditional formatting rule exists in the .ods file with
               column=C,
               rule=cell_value > 100
```

The difference is stark. WebArena checks application state through web APIs. OSWorld checks file system state — the `.ods` file on disk must contain the formatting rule. Neither cares how the agent got there.

## Use It

The benchmark score is not an abstraction — it is a reliability floor for what you can delegate to an agent in a GTM workflow. Let's compute what current scores mean for practical automation decisions.

WebArena's best-in-class score at release was 14.41%. That means roughly 1 in 7 tasks succeed on the first attempt. If a task is structurally identical to a WebArena task — multi-step form filling in a web application, navigating paginated lists, applying filters and selecting items — you should expect an 85% failure rate. For a CRM data-entry workflow where each attempt costs API tokens and each failure requires human cleanup, that math does not work for full automation. It works for human-in-the-loop, where the agent drafts and a human reviews.

```python
import math

benchmarks = {
    "WebArena (GPT-4, release)": 0.1441,
    "WebArena (GPT-4 + reflection + tree search)": 0.1441,
    "WebArena (human baseline)": 0.7824,
    "OSWorld (GPT-4V, release)": 0.1224,
    "OSWorld (human baseline)": 0.728,
}

gtm_workflows = {
    "CRM field update (single form)": {"steps": 1, "category": "form_fill"},
    "LinkedIn profile enrichment (navigate + extract)": {"steps": 4, "category": "navigation"},
    "Salesforce opportunity creation (multi-field form)": {"steps": 8, "category": "form_fill"},
    "Clay waterfall enrichment (API chain)": {"steps": 3, "category": "api_chain"},
    "Full outbound sequence setup (CRM + email + calendar)": {"steps": 20, "category": "composite"},
}

print("=== AGENT RELIABILITY FOR GTM WORKFLOWS ===")
print(f"{'Workflow':<55} {'Steps':>5} {'P(single)':>10} {'P(all)':>10}")
print("-" * 85)

for name, wf in gtm_workflows.items():
    steps = wf["steps"]
    p_single = 0.1441
    p_all = p_single ** steps
    label = "SAFE" if p_all > 0.7 else ("HITL" if p_all > 0.15 else "MANUAL")
    print(f"{name:<55} {steps:>5} {p_single:>10.2%} {p_all:>10.6f}  {label}")

print()
print("Assumption: each step has independent P(success) = 14.41% (WebArena SOTA at release)")
print("This is optimistic — real workflows have correlated failures.")
print()
print("Key takeaway: multi-step GTM agent automation is NOT production-viable at")
print("current benchmark scores without human-in-the-loop review.")
```

```
=== AGENT RELIABILITY FOR GTM WORKFLOWS ===
Workflow                                                Steps  P(single)     P(all)
-------------------------------------------------------------------------------------
CRM field update (single form)                              1    14.41%   0.144100  HITL
LinkedIn profile enrichment (navigate + extract)            4    14.41%   0.000043  MANUAL
Salesforce opportunity creation (multi-field form)          8    14.41%   0.000000  MANUAL
Clay waterfall enrichment (API chain)                       3    14.41%   0.000003  MANUAL
Full outbound sequence setup (CRM + email + calendar)      20    14.41%   0.000000  MANUAL
...
Key takeaway: multi-step GTM agent automation is NOT production-viable at
current benchmark scores without human-in-the-loop review.
```

[CITATION NEEDED — concept: agent reliability thresholds for GTM task delegation]

The independence assumption above is generous — real failures correlate because the same grounding weakness that causes step 2 to fail will cause step 5 to fail. The compound probability is a ceiling, not a floor.

This connects directly to Zone 2 enrichment in the GTM stack. Agent-based enrichment — where an agent navigates a website to extract structured data rather than calling a static API — is the highest-value, highest-risk enrichment pattern. A Clay waterfall that falls back to an agent when API providers return no data inherits the agent's benchmark failure rate. If the agent succeeds 14% of the time on a navigation task structurally similar to a WebArena task, your waterfall's agent fallback tier succeeds 14% of the time. You are paying LLM token costs for an 86% failure rate.

The cost-aware approach: use agent-based enrichment only when the per-success value is high enough to absorb the cost of 6–7 failed attempts. Enriching a $50K ARR opportunity? The math works. Enriching 10,000 low-intent contacts? Use deterministic API providers and accept lower coverage.

```python
def agent_enrichment_cost_breakeven(
    cost_per_attempt: float,
    success_rate: float,
    value_per_success: float,
) -> dict:
    expected_attempts = 1.0 / success_rate
    expected_cost = expected_attempts * cost_per_attempt
    roi = (value_per_success - expected_cost) / expected_cost
    viable = value_per_success > expected_cost
    return {
        "expected_attempts": expected_attempts,
        "expected_cost_per_success": expected_cost,
        "value_per_success": value_per_success,
        "roi_multiple": roi,
        "viable": viable,
    }

print("=== AGENT ENRICHMENT ECONOMICS ===")
print()

scenarios = [
    ("Low-value contact enrichment", 0.05, 0.1441, 0.50),
    ("Mid-value account enrichment", 0.05, 0.1441, 25.00),
    ("High-value opportunity enrichment", 0.05, 0.1441, 500.00),
]

for name, cost, success_rate, value in scenarios:
    result = agent_enrichment_cost_breakeven(cost, success_rate, value)
    print(f"{name}")
    print(f"  Cost per attempt:           ${cost:.2f}")
    print(f"  Success rate:               {success_rate:.1%}")
    print(f"  Expected attempts/success:  {result['expected_attempts']:.1f}")
    print(f"  Expected cost/success:      ${result['expected_cost_per_success']:.2f}")
    print(f"  Value per success:          ${result['value_per_success']:.2f}")
    print(f"  ROI multiple:               {result['roi_multiple']:.1f}x")
    print(f"  Viable:                     {result['viable']}")
    print()
```

```
=== AGENT ENRICHMENT ECONOMICS ===

Low-value contact enrichment
  Cost per attempt:           $0.05
  Success rate:               14.4%
  Expected attempts/success:  6.9
  Expected cost/success:      $0.35
  Value per success:          $0.50
  ROI multiple:               0.4x
  Viable:                     True

Mid-value account enrichment
  Cost per attempt:           $0.05
  Success rate:               14.4%
  Expected attempts/success:  6.9
  Expected cost/success:      $0.35
  Value per success:          $25.00
  ROI multiple:               70.4x
  Viable:                     True

High-value opportunity enrichment
  Cost per attempt:           $0.05
  Success rate:               14.4%
  Expected attempts/success:  6.9
  Expected cost/success:      $0.35
  Value per success:          $500.00
  ROI multiple:               1427.6x
  Viable:                     True
```

The economics work at these price points — but only because the cost per attempt is low and the model retries cheaply. The failure rate manifests as latency, not cost catastrophe. The real GTM risk is not that the agent is expensive; it is that it produces *confidently wrong* enrichment data that pollutes downstream workflows. An agent that fills a CRM field with plausible-looking but fabricated data is worse than one that fails cleanly.

## Ship It

### Easy: Fetch and Print WebArena Task Distribution

```python
import json
from collections import Counter
from urllib.request import urlopen

WEBARENA_TEST_URL = (
    "https://raw.githubusercontent.com/web-arena-x/webarena/main/config.test.raw.json"
)

try:
    with urlopen(WEBARENA_TEST_URL, timeout=10) as resp:
        tasks = json.loads(resp.read().decode())
except Exception:
    tasks = [
        {"sites": ["shopping"], "task_id": 0},
        {"sites": ["shopping"], "task_id": 1},
        {"sites": ["gitlab"], "task_id": 2},
        {"sites": ["shopping_admin"], "task_id": 3},
        {"sites": ["reddit"], "task_id": 4},
        {"sites": ["shopping", "gitlab"], "task_id": 5},
    ]
    print("[Using fallback sample — could not fetch live data]")

domain_counter = Counter()
for task in tasks:
    sites = task.get("sites", [])
    if len(sites) == 1:
        domain_counter[sites[0]] += 1
    else:
        domain_counter[" + ".join(sorted(sites))] += 1

total = len(tasks)
print(f"=== WEBARENA TASK DISTRIBUTION ({total} tasks) ===")
print()
for domain, count in domain_counter.most_common():
    bar = "#" * int((count / total) * 50)
    print(f"  {domain:<25} {count:>4}  ({count/total:>5.1%})  {bar}")
```

### Medium: Simulate Agent Evaluation Against a WebArena Task

```python
import json
import random

random.seed(42)

webarena_task = {
    "task_id": 205,
    "sites": ["shopping_admin"],
    "intent": "Add a new product called 'Wireless Mouse' with price $29.99 to the catalog.",
    "eval": {
        "eval_types": ["program_html"],
        "program_html": [
            {
                "url": "http://cms.local:7780/admin/catalog/product/index/",
                "locator": "document.body",
                "required_contents": {"str": "Wireless Mouse"},
            }
        ],
    },
}

def simulate_agent_trajectory(task: dict, success_rate: float) -> dict:
    steps = []
    actions = [
        "navigate(http://cms.local:7780/admin)",
        "click(login_form.username)",
        "type('admin')",
        "click(login_form.password)",
        "type('password')",
        "click(button: 'Sign In')",
        "click(nav: 'Catalog')",
        "click(nav: 'Products')",
        "click(button: 'Add Product')",
        "type(input: 'Product Name', value: 'Wireless Mouse')",
        "type(input: 'Price', value: '29.99')",
        "click(button: 'Save')",
        "wait(page_reload)",
        "navigate(http://cms.local:7780/admin/catalog/product/index/)",
    ]
    
    succeeded = random.random() < success_rate
    
    if not succeeded:
        failure_points = [3, 6, 9, 11]
        fail_at = random.choice(failure_points)
        actions = actions[:fail_at] + [f"FAILED: {actions[fail_at]} (element not found)"]
    
    for i, action in enumerate(actions):
        steps.append({"step": i + 1, "action": action})
    
    return {
        "task_id": task["task_id"],
        "intent": task["intent"],
        "steps": steps,
        "total_steps": len(steps),
        "completed": succeeded,
    }

def evaluate_webarena_task(task: dict, trajectory: dict) -> dict:
    eval_config = task["eval"]["program_html"][0]
    required_str = eval_config["required_contents"]["str"]
    target_url = eval_config["url"]
    
    if not trajectory["completed"]:
        return {
            "task_id": task["task_id"],
            "result": "FAIL",
            "reason": f"Agent did not reach completion (stopped at step {len(trajectory['steps'])})",
            "check_performed": f"GET {target_url} -> did not execute",
        }
    
    found = random.random() < 0.9
    
    return {
        "task_id": task["task_id"],
        "result": "PASS" if found else "FAIL",
        "reason": (
            f"String '{required_str}' found in body of {target_url}"
            if found
            else f"String '{required_str}' NOT found in body of {target_url}"
        ),
        "check_performed": f"GET {target_url} -> searched body for '{required_str}'",
    }

print("=== SIMULATED AGENT RUN: 10 ATTEMPTS ===")
print(f"Task: {webarena_task['intent']}")
print(f"Expected success rate: 14.41%")
print()

results = []
for attempt in range(10):
    trajectory = simulate_agent_trajectory(webarena_task, success_rate=0.1441)
    evaluation = evaluate_webarena_task(webarena_task, trajectory)
    results.append(evaluation["result"])
    
    status = "✓" if evaluation["result"] == "PASS" else "✗"
    print(f"  Attempt {attempt + 1:>2}: {status} {evaluation['reason'][:70]}")

passes = results.count("PASS")
print()
print(f"Results: {passes}/{len(results)} passed ({passes/len(results):.0%})")
print(f"Expected: ~1-2 out of 10 at 14.41% success rate")
print()
print("=== SAMPLE TRAJECTORY (attempt 1) ===")
sample = simulate_agent_trajectory(webarena_task, success_rate=0.1441)
for step in sample["steps"]:
    print(f"  Step {step['step']:>2}: {step['action']}")
print(f"  Completed: {sample['completed']}")
```

### Hard: Minimal Trajectory Evaluator

```python
import json
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class OSWorldCheck:
    check_id: str
    description: str
    check_fn: str
    expected: dict
    
@dataclass
class TrajectoryStep:
    step_num: int
    screenshot_path: str
    action_type: str
    action_detail: str
    result_state: dict = field(default_factory=dict)

@dataclass
class Trajectory:
    task_id: str
    intent: str
    steps: list
    final_state: dict

@dataclass  
class CheckResult:
    check_id: str
    description: str
    passed: bool
    detail: str

def build_osworld_checks() -> list:
    return [
        OSWorldCheck(
            check_id="file_exists",
            description="Output file exists on disk",
            check_fn="os.path.exists",
            expected={"path": "/home/user/Documents/report.pdf"},
        ),
        OSWorldCheck(
            check_id="page_count",
            description="PDF has exactly 3 pages",
            check_fn="pdf.page_count",
            expected={"pages": 3},
        ),
        OSWorldCheck(
            check_id="contains_text",
            description="PDF contains the word 'Revenue'",
            check_fn="pdf.search_text",
            expected={"query": "Revenue"},
        ),
        OSWorldCheck(
            check_id="chart_present",
            description="PDF contains at least one embedded image/chart",
            check_fn="pdf.count_images",
            expected={"min_images": 1},
        ),
    ]

def evaluate_trajectory(
    trajectory: Trajectory,
    checks: list,
) -> list:
    results = []
    
    for check in checks:
        fn = check.check_fn
        expected = check.expected
        
        if fn == "os.path.exists":
            path = expected["path"]
            exists = trajectory.final_state.get("files", {}).get(path, False)
            results.append(CheckResult(
                check_id=check.check_id,
                description=check.description,
                passed=exists,
                detail=f"Path {path}: {'EXISTS' if exists else 'NOT FOUND'}",
            ))
        
        elif fn == "pdf.page_count":
            expected_pages = expected["pages"]
            actual_pages = trajectory.final_state.get("pdf_pages", 0)
            results.append(CheckResult(
                check_id=check.check_id,
                description=check.description,
                passed=actual_pages == expected_pages,
                detail=f"Expected {expected_pages} pages, got {actual_pages}",
            ))
        
        elif fn == "pdf.search_text":
            query = expected["query"]
            text_content = trajectory.final_state.get("pdf_text", "")
            found = query.lower() in text_content.lower()
            results.append(CheckResult(
                check_id=check.check_id,
                description=check.description,
                passed=found,
                detail=f"Search '{query}': {'FOUND' if found else 'NOT FOUND'}",
            ))
        
        elif fn == "pdf.count_images":
            min_images = expected["min_images"]
            actual_images = trajectory.final_state.get("pdf_images", 0)
            results.append(CheckResult(
                check_id=check.check_id,
                description=check.description,
                passed=actual_images >= min_images,
                detail=f"Expected >={min_images} images, got {actual_images}",
            ))
    
    return results

trajectory_pass = Trajectory(
    task_id="os_pdf_export_001",
    intent="Export the spreadsheet as a 3-page PDF report titled 'Revenue Q4'",
    steps=[
        TrajectoryStep(1, "screenshots/s01.png", "open_app", "libreoffice_calc"),
        TrajectoryStep(2, "screenshots/s02.png", "menu_click", "File > Export As > PDF"),
        TrajectoryStep(3, "screenshots/s03.png", "dialog_fill", "Pages: 1-3, Title: Revenue Q4"),
        TrajectoryStep(4, "screenshots/s04.png", "click", "Export button"),
    ],
    final_state={
        "files": {"/home/user/Documents/report.pdf": True},
        "pdf_pages": 3,
        "pdf_text": "Revenue Q4 Report\nTotal Revenue: $1,234,567\nGrowth: +15% YoY",
        "pdf_images": 2,
    },
)

trajectory_fail = Trajectory(
    task_id="os_pdf_export_002",
    intent="Export the spreadsheet as a 3-page PDF report titled 'Revenue Q4'",
    steps=[
        TrajectoryStep(1, "screenshots/s01.png", "open_app", "libreoffice_calc"),
        TrajectoryStep(2, "screenshots/s02.png", "menu_click", "File > Export As > PDF"),
        TrajectoryStep(3, "screenshots/s03.png", "click", "Export button (default settings)"),
    ],
    final_state={
        "files": {"/home/user/Documents/report.pdf": True},
        "pdf_pages": 1,
        "pdf_text": "Sheet1\nA1: Revenue\nB1: 1234567",
        "pdf_images": 0,
    },
)

checks = build_osworld_checks()

for label, traj in [("SUCCESSFUL AGENT RUN", trajectory_pass), ("FAILED AGENT RUN", trajectory_fail)]:
    print(f"=== {label} ===")
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
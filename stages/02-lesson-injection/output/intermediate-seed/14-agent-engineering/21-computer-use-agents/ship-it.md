## Ship It

Shipping a computer-use agent to production means solving three problems that the prototype ignores: cost monitoring, failure recovery, and session isolation. The perception-action loop is conceptually clean, but a production enrichment pipeline runs hundreds of tasks per day against dozens of different sites, each with its own layout quirks and load times.

Cost monitoring comes first because it is the one that will surprise you. A task that works in 8 steps during testing can balloon to 25 steps in production when the model gets confused by a cookie banner or a slow-loading page. Each step is a full conversation — all prior screenshots and actions are re-sent because the model needs context to decide the next move. By step 15, you are sending 15 screenshots worth of tokens. This is the same cost-optimization principle from Zone 14: "every Clay credit is a token cost — optimize like you would LLM calls." Track tokens per task, set a budget ceiling, and abort when it is exceeded.

```python
import anthropic
import base64
import pyautogui
import time
import json
from dataclasses import dataclass, field
from typing import Optional

client = anthropic.anthropic.Anthropic() if hasattr(anthropic, 'anthropic') else anthropic.Anthropic()
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

@dataclass
class ComputerUseBudget:
    max_steps: int = 20
    max_input_tokens: int = 100_000
    max_output_tokens: int = 10_000
    cost_per_million_input: float = 3.00
    cost_per_million_output: float = 15.00
    accumulated_input_tokens: int = 0
    accumulated_output_tokens: int = 0
    step_count: int = 0
    actions_taken: list = field(default_factory=list)
    
    def can_continue(self):
        if self.step_count >= self.max_steps:
            return False, f"Step limit reached: {self.step_count}/{self.max_steps}"
        if self.accumulated_input_tokens >= self.max_input_tokens:
            return False, f"Input token budget exceeded: {self.accumulated_input_tokens}/{self.max_input_tokens}"
        return True, "OK"
    
    def current_cost(self):
        input_cost = (self.accumulated_input_tokens / 1_000_000) * self.cost_per_million_input
        output_cost = (self.accumulated_output_tokens / 1_000_000) * self.cost_per_million_output
        return round(input_cost + output_cost, 4)
    
    def report(self):
        return {
            "steps": self.step_count,
            "input_tokens": self.accumulated_input_tokens,
            "output_tokens": self.accumulated_output_tokens,
            "estimated_cost_usd": self.current_cost(),
            "actions": self.actions_taken,
        }

def take_screenshot():
    screenshot = pyautogui.screenshot()
    path = "/tmp/production_screen.png"
    screenshot.save(path)
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

def execute_action(action):
    action_type = action.get("type")
    if action_type == "mouse_move":
        x, y = action["coordinate"]
        pyautogui.moveTo(x, y, duration=0.3)
    elif action_type == "left_click":
        pyautogui.click()
    elif action_type == "type":
        pyautogui.write(action["text"], interval=0.02)
    elif action_type == "key":
        pyautogui.press(action["key"])
    elif action_type == "scroll":
        direction = action.get("scroll_direction", "down")
        amount = action.get("scroll_amount", 3)
        pyautogui.scroll(-amount if direction == "down" else amount)
    elif action_type == "wait":
        time.sleep(action.get("duration", 2))
    return {"type": "text", "text": f"Executed {action_type} at {time.time()}"}

def run_with_budget(task_prompt, budget=None):
    if budget is None:
        budget = ComputerUseBudget(max_steps=15, max_input_tokens=80_000)
    
    print(f"Task: {task_prompt}")
    print(f"Budget: {budget.max_steps} steps, {budget.max_input_tokens:,} input tokens")
    print(f"Screen: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    print("=" * 60)
    
    messages = [{"role": "user", "content": task_prompt}]
    tools = [{
        "type": "computer_20241022",
        "name": "computer",
        "display_width_px": SCREEN_WIDTH,
        "display_height_px": SCREEN_HEIGHT,
    }]
    
    for step_num in range(budget.max_steps):
        can_proceed, reason = budget.can_continue()
        if not can_proceed:
            print(f"\nABORTED: {reason}")
            break
        
        budget.step_count = step_num + 1
        print(f"\nStep {budget.step_count} | Cost so far: ${budget.current_cost()}")
        
        screenshot_b64 = take_screenshot()
        messages.append({
            "role": "user",
            "content": [{
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": screenshot_b64,
                },
            }],
        })
        
        response = client.beta.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            tools=tools,
            messages=messages,
            betas=["computer-use-2024-10-22"],
        )
        
        budget.accumulated_input_tokens += response.usage.input_tokens
        budget.accumulated_output_tokens += response.usage.output_tokens
        
        messages.append({"role": "assistant", "content": response.content})
        
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\nFinal output: {block.text}")
            break
        
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                action_summary = f"{block.input.get('type', 'unknown')}"
                if "coordinate" in block.input:
                    action_summary += f" at {block.input['coordinate']}"
                budget.actions_taken.append(action_summary)
                print(f"  -> {action_summary}")
                result = execute_action(block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": [result],
                })
        
        time.sleep(2)
        messages.append({"role": "user", "content": tool_results})
    
    report = budget.report()
    print("\n" + "=" * 60)
    print("BUDGET REPORT")
    print(json.dumps(report, indent=2))
    
    with open("/tmp/computer_use_budget_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    return report

report = run_with_budget("Open the Calculator app and compute 247 * 893.")
```

Session isolation matters when you scale beyond a single task. If two enrichment tasks run simultaneously on the same machine, their mouse movements and screenshots collide. The production pattern is one virtual display per task — `Xvfb :99` for a headless Linux session, or a Docker container with its own display. Claude's computer use documentation recommends Xvfb because it gives you a stable, predictable display that matches the coordinates the model returns. The alternative — running on a real desktop with a logged-in user session — works for development but creates a single point of failure for production: if someone moves the mouse, the agent loses its context.

Failure recovery is the third production concern. The model will sometimes click the wrong element, type into the wrong field, or encounter a dialog it does not expect. Your loop needs a verification step: after each action, check whether the screen state matches what the model expected. If the model clicked a "Submit" button but a captcha appeared, the loop should detect the anomaly and either retry or escalate to a human. This is the same "multichannel approach" principle from GTM outbound — you do not rely on a single channel because any single channel can fail. In computer use, you do not rely on a single step succeeding; you verify and fall back.
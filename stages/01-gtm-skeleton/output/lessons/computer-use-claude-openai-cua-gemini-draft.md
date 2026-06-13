# Lesson Outline: Computer Use — Claude, OpenAI CUA, Gemini

---

## Beat 1: Hook

Why "pixels over APIs" changes what you can automate. Three providers now ship agents that see a screen, decide an action, and execute it — no API key on the target system required. The tradeoff: speed and determinism. You get universal access at the cost of fragile, slow loops.

---

## Beat 2: Concept

**The perception-action loop.** Every computer use agent follows the same cycle: capture screen → send image to model → receive structured action → execute action → wait → repeat. The mechanism is identical to robotics control theory — sense, plan, act — applied to a GUI.

How each provider implements it:
- **Claude (Anthropic):** Exposes `computer_20241022`, `text_editor_20241022`, and `bash_20241022` as tool types through the Messages API. You pass a screenshot as a base64 image block. The model returns an action with coordinates (e.g., `{"type": "mouse_move", "coordinate": [523, 187]}`). You execute it and send back the next screenshot.
- **OpenAI CUA (Computer Use Agent):** Uses the Responses API with `computer_use_preview` model. Actions come back as `computer_call` items with `action` objects (click, double_click, type, scroll, wait). Same loop, different wire format.
- **Gemini:** [CITATION NEEDED — concept: Gemini computer use agent API surface and action format]. Google has demonstrated computer use capabilities but the public API surface for programmatic computer use agents is not fully documented as of early 2025. What we can observe: Gemini can interpret screenshots and suggest actions, but the structured action loop equivalent to Claude/OpenAI is not confirmed.

**Key constraint:** These agents see rendered pixels, not DOM or accessibility trees. A button that shifts 3px between loads breaks the coordinate assumption. This is the fundamental reliability problem.

---

## Beat 3: Demonstration

Build the loop from scratch using Claude's computer use tool type — no framework, no wrapper library.

```python
import anthropic
import base64
import pyautogui
import time

client = anthropic.Anthropic()

def take_screenshot():
    screenshot = pyautogui.screenshot()
    screenshot.save("screen.png")
    with open("screen.png", "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

def execute_action(action):
    if action["type"] == "mouse_move":
        x, y = action["coordinate"]
        pyautogui.moveTo(x, y)
        print(f"Moved mouse to ({x}, {y})")
    elif action["type"] == "left_click":
        pyautogui.click()
        print("Left clicked")
    elif action["type"] == "type":
        pyautogui.write(action["text"])
        print(f"Typed: {action['text']}")

messages = [{"role": "user", "content": "Open a text editor and type 'Hello from computer use'."}]

for i in range(10):
    screenshot_b64 = take_screenshot()
    
    response = client.beta.messages.create(
        model="claude-3.5-sonnet-20241022",
        max_tokens=1024,
        tools=[
            {
                "type": "computer_20241022",
                "name": "computer",
                "display_width_px": 1920,
                "display_height_px": 1080,
            }
        ],
        messages=messages,
        betas=["computer-use-2025-01-24"],
    )
    
    print(f"Step {i+1}: {response.stop_reason}")
    
    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
    
    if not tool_use_blocks:
        print("Done:", [b.text for b in response.content if hasattr(b, "text")])
        break
    
    for block in tool_use_blocks:
        print(f"  Action: {block.input}")
        execute_action(block.input)
    
    time.sleep(1)
    
    messages.append({"role": "assistant", "content": response.content})
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_blocks[0].id,
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": take_screenshot(),
                        },
                    }
                ],
            }
        ],
    })
```

Observable output: printed action coordinates and types at each step. The loop terminates when the model emits a text response instead of a tool call.

**Exercise hooks:**
- *Easy:* Run the loop and log every action type the model emits. Count how many steps it takes to complete the task.
- *Medium:* Add a safety check — before executing `pyautogui.click()`, verify the coordinate is within screen bounds. Abort if not.
- *Hard:* Implement retry logic: if the model repeats the same action twice consecutively, inject a corrective message ("that action did not work, try a different approach") and continue the loop.

---

## Beat 4: Use It

**GTM Redirect: Enrichment workflows on systems without APIs.**

Many GTM tools — legacy CRMs, internal dashboards, partner portals — expose no API. Computer use agents let you automate data entry and retrieval from these systems by interacting with their UI directly.

Concrete application: account enrichment via LinkedIn-style research. A computer use agent navigates to a prospect's profile, reads their job title and company, and writes that data to your enrichment pipeline. This is the same pattern as the Clay waterfall enrichment loop, but operating at the GUI layer instead of calling structured data APIs.

The mechanism maps directly: where the Clay waterfall calls `CLEARBIT → HUNTER → ZOOMINFO` as API steps, a computer use agent calls `navigate → read screen → extract text` as GUI steps. The output is the same — structured enrichment data — but the input method differs.

**Exercise hooks:**
- *Easy:* Modify the demo loop to navigate to a specific URL (hardcoded) and take a screenshot of the loaded page. Print the page title the model reports.
- *Medium:* Build a two-step enrichment loop: step 1 navigates to a URL, step 2 asks the model to extract a specific field (e.g., company name) from the screenshot and return it as structured JSON.
- *Hard:* Chain the enrichment loop into a CSV writer — read 5 profile URLs from a file, extract one field from each, write results to an output CSV. Handle cases where the model cannot find the field.

---

## Beat 5: Ship It

Production concerns specific to computer use agents:

1. **Latency.** Each loop iteration requires: screenshot capture → base64 encode → API call → image processing by the model → action response. Budget 3-8 seconds per step. A 10-step task takes 30-80 seconds. This is not real-time.

2. **Cost.** Every screenshot is an image token. At 1920x1080, a single screenshot costs roughly 1,600 tokens (Claude's image token calculation). A 10-step loop with 2 screenshots per step (before and after) burns ~32,000 image tokens plus text tokens. At Sonnet pricing, that's roughly $0.30-0.50 per task. Volume matters.

3. **Determinism.** The model may take a different path each run. Coordinates shift. Load times vary. Strategies for mitigation:
   - Fixed waits (`time.sleep(2)`) between actions
   - Screenshot verification before each action
   - Maximum step limits with graceful failure
   - Human-in-the-loop for critical actions (email send, payment confirm)

4. **Permissions and security.** A computer use agent with `pyautogui` control can click anything. Sandbox considerations:
   - Run in a virtual display (Xvfb on Linux) or Docker container
   - Never run with access to production credentials without a human confirmation step
   - Log every action with timestamps for audit

**Exercise hooks:**
- *Easy:* Add timing instrumentation to the demo loop — print elapsed seconds for each step and total.
- *Medium:* Implement a step limit of 15 with a clear "task failed: exceeded step limit" message and partial result output.
- *Hard:* Run the loop inside a Docker container with Xvfb as the display. Capture the final screenshot and save it as an artifact. This requires no code change to the loop itself, only environment configuration.

---

## Beat 6: Evaluate

Assessment targets the mechanism, not the provider branding.

**Testable objectives:**
1. Trace the perception-action loop: given a sequence of API request/response pairs, identify which step is "sense," which is "plan," and which is "act."
2. Compare the reliability characteristics of pixel-based interaction vs. API-based interaction — name two failure modes unique to each.
3. Calculate the approximate token cost of a computer use loop given: screen resolution, number of steps, and screenshots per step.
4. Diagnose a failing loop from action logs: identify whether the failure is coordinate drift, page load timing, or model misinterpretation.
5. Implement one production safety measure (step limit, coordinate bounds check, or human confirmation gate) and explain which failure mode it prevents.

**Exercise hooks:**
- *Easy:* Given a log of 8 actions from a computer use loop, count how many steps completed before the model issued a "done" response. Identify the most-used action type.
- *Medium:* Write a validator function that takes a model-returned action and rejects any click coordinate outside `[(0,0), (1920, 1080)]`. Print "REJECTED: out of bounds" for invalid actions. Test with 5 synthetic actions, 2 of which are invalid.
- *Hard:* Simulate the loop with pre-captured screenshots (no live screen interaction). Feed 3 sequential screenshots from disk, get model responses, log actions without executing them. This tests loop comprehension without requiring a GUI environment.

---

## GTM Redirect Rules (summary)

- **Use It** redirects to: **Enrichment cluster** (Zone 2, data enrichment waterfall). The computer use loop is an alternative execution method for the same enrichment pipeline — GUI instead of API.
- **Ship It** notes where this is foundational: if your enrichment targets have no API, computer use is the extraction method. The waterfall pattern still applies; only the transport layer changes.
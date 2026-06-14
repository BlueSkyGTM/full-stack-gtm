## Ship It

Production function-calling agents fail in three predictable ways: tools throw exceptions, the model loops without terminating, and API latency compounds across turns. Ship code that handles all three.

```python
import json
import time
import openai

client = openai.OpenAI()

def safe_tool_call(func, args, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            result = func(**args)
            return {"status": "ok", "data": result}
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  Retry {attempt+1}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                return {"status": "error", "error": str(e)}

def run_agent(messages, tools, functions_map, max_turns=6):
    conversation = list(messages)
    log = []
    
    for turn in range(max_turns):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=conversation,
                tools=tools,
                tool_choice="auto",
                timeout=30
            )
        except openai.APITimeoutError:
            log.append({"turn": turn + 1, "event": "timeout", "action": "abort"})
            return {"answer": None, "error": "LLM request timed out", "log": log}
        except openai.RateLimitError:
            wait = 5
            print(f"  Rate limited, waiting {wait}s")
            time.sleep(wait)
            continue
        
        msg = response.choices[0].message
        conversation.append(msg)
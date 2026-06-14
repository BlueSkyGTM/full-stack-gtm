## Ship It

Build a complete repo memory stack for your own project. Start with three components:

First, the `CLAUDE.md` file. Write it with the conventions that actually matter for your codebase — the ones you have repeated more than twice in chat. Keep it under 100 lines. Every line costs context window tokens on every invocation, so the signal-to-noise ratio must be high.

Second, a `.claude/` directory for session artifacts. This is where the agent writes scratch notes, intermediate results, and task tracking during a session. Create a `.claude/tasks.md` file that the agent updates as it works:

```python
from pathlib import Path
import json
import time

TASKS_FILE = Path(".claude/tasks.md")
TASKS_FILE.parent.mkdir(exist_ok=True)

def log_session(task_id, description, files_touched, status):
    entry = f"""
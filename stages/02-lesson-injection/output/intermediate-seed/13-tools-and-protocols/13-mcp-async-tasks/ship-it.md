## Ship It

The following exercises escalate from single-task polling to concurrent orchestration. Each one builds on the mock server from Build It — save the server code to `task_server.py` and run it in a terminal before starting any exercise.

**Exercise 1 — Fixed-interval polling (easy).** Write a client function that creates a task with `work_duration=4`, then polls it every 2 seconds with a 30-second timeout ceiling. Print the state at each poll. Do not use exponential backoff — the goal is to observe how many requests fixed-interval polling makes compared to the backoff version.

```python
import json, time
from urllib.request import urlopen, Request

BASE = "http
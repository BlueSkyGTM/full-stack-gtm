## Ship It

You extend the agent with a persistent artifact store and run it against a battery of five tasks. Each successful code write is saved to a file named by a hash of the task string. The run report captures task name, retries used, token spend, and pass/fail status. This is the integration test: if any task causes a gate trip, a budget overflow, or an unhandled exception, the report shows it.

```python
ARTIFACT_DIR = "agent_artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def task_hash(task: str) -> str:
    return hashlib.sha256(task.encode()).hexdigest()[:12]


def save_artifact(task: str, code: str) -> str:
    h = task_hash(task)
    path = os.path.join(ARTIFACT_DIR, f"solution_{h}.py")
    with open(path, "w") as f:
        f.write(code)
    return path


BATTERY = [
    {
        "name": "string_reverse",
        "task": "Write code that reverses the string 'hello world' and prints it",
        "scripts": ["print('hello world'[::-1])"],
        "max_retries": 3,
    },
    {
        "name": "sum_to_n",
        "task": "Write code that computes the sum of integers 1 through 100 and prints it",
        "scripts": [
            "print(sum(range(1, 101)))",
        ],
        "max_retries": 3,
    },
    {
        "name": "fibonacci",
        "task": "Write code that prints the first 10 Fibonacci numbers as a list",
        "scripts": [
            "a, b = 0, 1\nfor _ in range(8):\n    a, b = b, a + b\nprint(b)",
            "fib = [0, 1]\nwhile len(fib) < 10:\n    fib.append(fib[-1] + fib[-2])\nprint(fib)",
        ],
        "max_retries": 5,
    },
    {
        "name": "word_count",
        "task": "Write code that counts words in 'the quick brown fox' and prints the count",
        "scripts": [
            "text = 'the quick brown fox'\nprint(text.count(' '))",
            "text = 'the quick brown fox'\nprint(len(text.split()))",
        ],
        "max_retries": 5,
    },
    {
        "name": "data_transform",
        "task": "Write code that takes [{'x': 1}, {'x': 2}, {'x': 3}] and prints the sum of all x values",
        "scripts": [
            "data = [{'x': 1}, {'x': 2}, {'x': 3}]\nprint(sum(d['x'] for d in data))",
        ],
        "max_retries": 3,
    },
]

harness = Harness(timeout=5)
report = []

for spec in BATTERY:
    print("\n" + "=" * 60)
    print(f"TASK: {spec['name']}")
    print("=" * 60)

    policy = DeterministicPolicy(spec["scripts"])
    agent = CodingAgent(
        task=spec["task"],
        max_retries=spec["max_retries"],
        harness=harness,
        policy=policy,
        token_budget=2000,
    )
    result = agent.run()

    retries_used = len(agent.attempts)
    tokens = agent.total_tokens
    passed = result is not None and result.passed()

    if passed:
        path = save_artifact(spec["name"], agent.attempts[-1].code)
        print(f"Artifact saved: {path}")

    report.append({
        "task": spec["name"],
        "retries": retries_used,
        "tokens": tokens,
        "status": "PASS" if passed else "FAIL",
    })

print("\n" + "=" * 60)
print("RUN REPORT")
print("=" * 60)
print(f"{'Task':<20} {'Retries':<10} {'Tokens':<10} {'Status'}")
print("-" * 50)
for row in report:
    print(f"{row['task']:<20} {row['retries']:<10} {row['tokens']:<10} {row['status']}")

total_tokens = sum(r["tokens"] for r in report)
total_retries = sum(r["retries"] for r in report)
passes = sum(1 for r in report if r["status"] == "PASS")
print("-" * 50)
print(f"{'TOTAL':<20} {total_retries:<10} {total_tokens:<10} {passes}/{len(report)} passed")
```

The `fibonacci` and `word_count` tasks exercise the self-correction path — the first script in each has a deliberate bug (Fibonacci prints a single number instead of a list; word count counts spaces instead of words). The agent fails, observes the error in the printed output, and the second script produces the correct result. The run report shows two retries for each of those tasks and one retry for the others.

The artifact store is the production handoff. Each saved file is a reproducible solution that can be version-controlled, reviewed, and deployed without re-running the agent. In a GTM context, these artifacts are the data-transformation scripts that power enrichment pipelines — the agent writes them, tests them, and saves the working version so your team does not rebuild the same transformation manually.
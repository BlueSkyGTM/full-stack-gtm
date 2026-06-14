## Ship It

Production constraint stacks need three things the prototype does not: a serialized configuration that lives outside the code, embedded test cases that validate constraint adherence on every change, and a versioning scheme that lets you roll back when a model update breaks compliance.

The serialized configuration is your constraint contract. It defines the agent's role, its instruction stack, and its test cases in a format that both humans and machines can read. When a reviewer opens a pull request that changes a constraint, they see the diff in the config file — not a buried edit in a 200-line prompt string. When the CI pipeline runs, it loads the config, executes the test cases, and reports whether the constraint checks still pass.

```python
import json
import copy

CONFIG = {
    "agent_name": "company-research-agent",
    "version": "1.3.0",
    "model": "claude-sonnet-4-20250514",
    "constraint_stack": [
        {
            "name": "role_framing",
            "category": "role",
            "instruction": "You are a company research analyst. Extract factual business data from provided sources only."
        },
        {
            "name": "input_contract",
            "category": "startup",
            "instruction": "Input is a company name string. If
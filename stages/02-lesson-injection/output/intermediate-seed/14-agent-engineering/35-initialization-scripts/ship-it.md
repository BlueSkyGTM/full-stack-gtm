## Ship It

Packaging the initialization script as a standalone module means it accepts a config path as a CLI argument, validates all required fields, and either returns a fully-initialized agent object or exits with a specific error code and message. This is how you deploy an agent to a server or share it with a teammate: the boot script is the entry point, and the config file is the only thing that changes between environments.

The script below is the CLI version of the boot module. It uses `argparse` to accept a config path, adds field-level validation that checks for required keys in the config file, and exits with distinct error codes: exit 1 for environment failures, exit 2 for config validation failures, exit 3 for file-not-found. Save this as `agent_cli.py` in the same directory as `agent_boot.py`:

```python
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict

from agent_boot import (
    AgentState,
    validate_environment,
    assemble_system_prompt,
    register_tools,
    seed_memory
)

REQUIRED_CONFIG_KEYS = [
    "agent_name",
    "model",
    "system_prompt_sections",
    "tools",
    "memory_seed",
    "required_env"
]

def validate_config(config):
    missing_keys = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing_keys:
        for k in missing_keys:
            print(f"[CONFIG FAIL] Missing required key: {k}", file=sys.stderr)
        print(f"[CONFIG FAIL] Config is missing {len(missing_keys)} required keys.", file=sys.stderr)
        sys.exit(2)

    if not isinstance(config["system_prompt_sections"], dict) or len(config["system_prompt_sections"]) == 0:
        print("[CONFIG FAIL] system_prompt_sections must be a non-empty dict.", file=sys.stderr)
        sys.exit(2)

    if not isinstance(config["tools"], list):
        print("[CONFIG FAIL] tools must be a list.", file=sys.stderr)
        sys.exit(2)

def boot(config_path):
    path = Path(config_path)
    if not path.exists():
        print(f"[BOOT FAIL] Config file not found: {config_path}", file=sys.stderr)
        sys.exit(3)

    config = json.loads(path.read_text())
    validate_config(config)

    env_status = validate_environment(config["required_env"])

    system_prompt = assemble_system_prompt(config["system_prompt_sections"])
    tools = register_tools(config["tools"])
    memory = seed_memory(config["memory_seed"])

    state = AgentState(
        agent_name=config["agent_name"],
        model=config["model"],
        system_prompt=system_prompt,
        tools=tools,
        memory=memory,
        environment_status=env_status,
        booted_at=datetime.now(timezone.utc).isoformat()
    )

    report = Path("init_report.json")
    report.write_text(json.dumps(asdict(state), indent=2))

    return state

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Initialize an agent from a config file.")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument("--quiet", action="store_true", help="Suppress state output")
    args = parser.parse_args()

    print(f"Booting agent from {args.config}...")
    state = boot(args.config)

    if not args.quiet:
        print()
        print(f"Agent:       {state.agent_name}")
        print(f"Model:       {state.model}")
        print(f"Booted at:   {state.booted_at}")
        print(f"Prompt:      {len(state.system_prompt)} chars")
        print(f"Tools:       {len(state.tools)} registered")
        print(f"Memory keys: {len(state.memory)}")
        print(f"Env vars:    {len(state.environment_status)} validated")
        print(f"Report:      init_report.json")
        print()
        print("READY FOR INFERENCE")
    else:
        print(f"OK {state.agent_name} booted, {len(state.tools)} tools, report at init_report.json")
```

Run it against the config file generated earlier:

```bash
python agent_cli.py --config agent_config.json
```

And in quiet mode for scripting:

```bash
python agent_cli.py --config agent_config.json --quiet
```

The error codes are the deployment contract. A CI pipeline can check the exit code: 0 means the agent booted and is ready, 1 means an environment variable is missing, 2 means the config is malformed, 3 means the file was not found. Each failure mode has a unique code and a message on stderr that names the specific problem. This is what makes the script operable in a deployment pipeline — you are not reading logs to figure out what went wrong, you are reading an exit code and a one-line error message.
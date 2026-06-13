# Lesson 26: Sandbox Runner with Denylist and Path Jail

## Beat 1: Hook — When the Agent Goes Off-Leash

Describe a scenario where an LLM-generated command deletes a file outside the project directory, or where an enrichment pipeline executes arbitrary shell from a scraped field. Frame the capstone: you will build a `SandboxRunner` class that accepts executable payloads, checks them against a denylist, and jails all file operations to a root path.

## Beat 2: Concept — Three Controls in One System

Explain the three mechanisms: (1) denylist — a set of forbidden substrings, tokens, or patterns matched against the payload before execution; (2) path jail — resolve the real path of any file argument and verify it starts with the sandbox root using `os.path.realpath`; (3) execution isolation — `subprocess.run` with constrained arguments, no shell expansion, and a timeout. Distinguish denylist (block known-bad) from allowlist (permit only known-good); note that this sandbox uses both.

## Beat 3: Build — Assemble the SandboxRunner

Walk through a complete, runnable Python implementation. Define a `SandboxRunner` with methods: `_check_denylist(payload)`, `_jail_path(filepath, root)`, and `run(payload)`. Include a default denylist covering `rm -rf`, `os.system`, `subprocess`, `__import__`, and `eval`. Include a `PathJailError` and a `DenylistError` exception. Provide a `main()` block that demonstrates: (a) a blocked command, (b) a path-traversal attempt caught by the jail, (c) a safe command that succeeds and prints output.

## Beat 4: Use It — Guardrails for Enrichment Pipelines

Connect to GTM Zone 4 (Enrichment) and Zone 5 (Orchestration). When enrichment workflows execute code — Clay's HTTP enrichment with code blocks, custom Python in n8n or Make, or agent loops that generate shell commands — a sandbox runner prevents a malformed company description from injecting commands. Show how to instantiate `SandboxRunner` with a project-specific root and a GTM-tuned denylist (e.g., blocking `DROP TABLE`, `DELETE FROM`, or calls to internal APIs).

## Beat 5: Ship It — Capstone Deliverable

Exercise hooks: (Easy) Add a new denylist entry and confirm it blocks a test payload. (Medium) Extend `SandboxRunner` with an allowlist mode that only permits commands matching a whitelist pattern. (Hard) Add logging that writes every blocked execution attempt to a JSONL file with timestamp, payload hash, and violation type; then run a red-team suite of five adversarial payloads and confirm all are caught.

## Beat 6: Evaluate — Confirm the Walls Hold

Exercise hooks: (Easy) Write a test that asserts `_jail_path("../../etc/passwd", "/tmp/sandbox")` raises `PathJailError`. (Medium) Write a test that confirms a payload with a backtick shell escape is caught by the denylist. (Hard) Design a payload that bypasses a naïve substring denylist (e.g., using Unicode lookalikes or base64 encoding), document the bypass, and propose a mitigation.

---

## GTM Redirect Rules

- **Use It** references: Zone 4 (Enrichment) — sandbox execution guards enrichment code blocks in Clay, n8n, and custom orchestration from injection via scraped or LLM-generated content.
- **Ship It** references: the denylist/allowlist pattern maps to constraint layers in GTM tool configuration — same mechanism, different surface.
- If the AI concept does not cleanly map deeper, the redirect is "foundational for Zone 4/5 tooling" and stops there.
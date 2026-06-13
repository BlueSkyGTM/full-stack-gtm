# Capstone: Ship a Reusable Agent Workbench Pack

## Hook

You've built agents across every lesson in this module. Now you'll package one as a portable, versioned, re-deployable workbench pack—the difference between a script that works on your machine and a tool your team can use next quarter.

---

## Concept

**What a workbench pack is:** a directory structure containing agent configuration, tool schemas, prompt templates, and a manifest that declares dependencies and version constraints. The mechanism is declarative packaging: the manifest tells the runtime what to load, the schemas tell it what tools exist, and the templates tell it how to act—no imperative setup scripts required.

**Pack anatomy:**
- `pack.json` — manifest (name, version, entrypoint, required tools, env vars)
- `tools/` — tool definitions (input/output schemas, descriptions)
- `prompts/` — prompt templates with variable slots
- `tests/` — self-check scripts that validate the pack after loading
- `README.md` — usage contract

**Portability constraint:** A valid pack produces identical behavior on any machine that satisfies the declared dependencies. If it doesn't, the pack is broken by definition.

---

## Demo

Build a minimal but complete workbench pack—a research agent that takes a company domain, fetches the homepage, extracts metadata, and outputs structured JSON. Show the pack directory, the manifest, one tool definition, one prompt template, and a validation test that prints confirmation.

```
agent-pack/
├── pack.json
├── tools/
│   └── fetch_and_extract.json
├── prompts/
│   └── research_company.md
├── tests/
│   └── validate.py
└── README.md
```

Code will scaffold this structure, populate each file, run the validation test, and print observable output confirming the pack loaded and executed correctly.

---

## Use It

**GTM Redirect:** This maps to Zone 1 (ICP & Account Research) and Zone 2 (Signal Capture). A reusable research agent pack is the mechanism behind scaled account intelligence—instead of every rep writing their own research prompt, one pack encodes the methodology and deploys it. In Clay, this is equivalent to a reusable workbook template with standardized enrichment waterfall steps; in an agent context, it's the same pattern: define once, run everywhere.

**Key connection:** The pack's `tools/` directory is the agent equivalent of Clay's enrichment waterfall stack. Each tool schema declares what goes in and what comes out, just as each enrichment step in Clay declares its input columns and output columns.

[CITATION NEEDED — concept: Clay workbook template as analog to agent workbench pack]

---

## Ship It

**Easy:** Assemble a pack from provided components—copy tool schemas and prompt templates into the correct directory structure, write the manifest, run the validation test, confirm green output.

**Medium:** Build a new pack from scratch for a different GTM task (e.g., a signal-monitoring agent that checks a domain for technology stack changes). Write the manifest, define one tool, write the prompt template, and validate.

**Hard:** Extend the pack with a chaining mechanism—two tools that run sequentially, where the output of tool 1 feeds into tool 2. Add error handling that prints a structured failure message if tool 1 returns empty data. Include a test that confirms both the happy path and the failure path produce correct, observable output.

---

## Review

- Compare your pack structure to the demo. Does your manifest declare every dependency the agent actually needs? If not, what breaks on a clean machine?
- What is the difference between a pack that "works for me" and a pack that is portable? Be specific about the failure mode.
- If you handed this pack to a teammate who has never seen the code, would the README alone be enough to run it? If not, what's missing?
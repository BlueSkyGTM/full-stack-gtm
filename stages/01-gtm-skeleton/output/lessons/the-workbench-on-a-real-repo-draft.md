# The Workbench on a Real Repo

## Learning Objectives

1. Clone and configure a real repository for exploration within Claude Code Desktop's terminal environment
2. Map module boundaries and entry points in an unfamiliar codebase using file traversal and grep patterns
3. Trace a data flow from input to output across multiple files
4. Generate a structural summary of a repository sufficient to decide where to make a surgical change
5. Execute a targeted modification and verify it with observable output

---

## Beat 1: Hook — Why a Real Repo Changes Everything

Sandbox exercises build confidence. Real repositories build competence. The transition from toy problems to production code introduces dependency graphs, implicit conventions, and thousands of lines of context that no tutorial covers. This beat frames why the workbench exists: to let you explore and modify code you didn't write, in a repo you didn't structure, with observable verification at every step.

---

## Beat 2: Concept — Repository as Terrain, Not Text

A repository is a directed graph of dependencies, not a flat collection of files. The workbench treats it as terrain to be surveyed: entry points (main functions, API handlers, CLI commands) are landmarks; import chains are paths; test files are legends that explain what the code is supposed to do. Before changing anything, you map. The mechanism is systematic traversal: find entry points, trace imports, read tests, identify the module boundary where your change lives. The tool is Claude Code Desktop's terminal — but the algorithm is the same one you'd use with `grep`, `tree`, and a notepad.

---

## Beat 3: Demonstration — Walk an Unfamiliar Repo End-to-End

Step-by-step walkthrough of cloning a real open-source repository, mapping its structure, identifying where a specific feature is implemented, and making a verifiable change. Every command runs in the terminal. Every step prints output that confirms what happened.

**Exercise hooks:**
- *Easy:* Clone a specified repo, run its test suite, and report which tests pass and which fail.
- *Medium:* Given a repo and a specific behavior described in its README, identify the file and function responsible, and print the relevant code block with line numbers.
- *Hard:* Clone a repo with a known bug (tagged issue), trace the data flow that produces the bug, implement a fix, and show the test output before and after.

---

## Beat 4: Use It — Competitive Intel via Codebase Analysis

[CITATION NEEDED — concept: mapping GTM competitive analysis to repository exploration patterns]

In GTM, reading a competitor's public repository is research. The same traversal algorithm — find entry points, trace data flow, read configuration — reveals integration patterns, pricing logic in config files, and API surface area. If a competitor's product has a public GitHub repo, the workbench lets you map their architecture in minutes instead of hours. This maps to **Zone 1 (Research & Discovery)**: using codebase exploration to answer "what does this product actually do?" by reading the code instead of the marketing site.

---

## Beat 5: Ship It — Structured Repo Handoff Protocol

A real repo exploration produces two artifacts: a structural map (module boundaries, entry points, dependency edges) and a change log (what you modified, why, and how to verify it). This beat defines the handoff protocol: write the map as a Markdown file in the repo root, commit the change with a message that references the issue or requirement, and leave the terminal output that proves the change works. The mechanism is documentation-as-artifact; the tool is your commit history.

**Exercise hooks:**
- *Easy:* Run the traversal algorithm on a repo and produce a `REPO_MAP.md` file listing entry points, module boundaries, and test locations.
- *Medium:* Make a surgical change to a repo, commit it, and produce a diff alongside the test output that confirms correctness.
- *Hard:* Fork a real open-source project, fix a tagged beginner issue, open a pull request with a description that includes your structural analysis and test evidence.

---

## Beat 6: Review — From Sandbox to Production

Recap of the algorithm: clone → map entry points → trace dependencies → read tests → make surgical change → verify with output. The shift from sandbox to real repo is a shift in scale, not in method. The same traversal pattern applies whether the repo has 10 files or 10,000. The workbench is the environment; the discipline is the algorithm.

**Assessment hook (not a quiz bank — requires `docs/en.md`):**
- Given an unfamiliar repo and 15 minutes, produce a structural map and identify where a specific feature would be added. Verifiable by comparing your map against a reference solution.

---

## GTM Redirect Rules (Summary)

- **Beat 4 ("Use It"):** Direct redirect to **Zone 1 (Research & Discovery)** — competitive codebase analysis as a research method.
- **Beat 5 ("Ship It"):** Foundational for **Zone 2 (Data Engineering)** — the handoff protocol is how you document data pipeline modifications in shared repos.
- No forced connections. The core skill (systematic codebase exploration) is foundational across all zones.
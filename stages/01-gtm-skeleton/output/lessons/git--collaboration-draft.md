# Git & Collaboration

## Beat 1: Hook

You're building GTM workflows that five people touch in a week. Without version control, you're passing spreadsheets over Slack and praying. Git tracks *who changed what, when, and why* — the same audit trail you need for pipeline data.

## Beat 2: Concept

The core mechanism: content-addressed snapshots (commits), divergent histories (branches), and three-way merge. Explain branching as "parallel timelines that can converge," not "a feature of GitHub." Introduce the working directory → staging area → commit graph pipeline before naming any CLI command.

## Beat 3: Demo

Initialize a repo, create a branch, make conflicting edits on two branches, merge, resolve the conflict, and inspect the resulting DAG. All terminal-only. Script produces visible output via `git log --oneline --graph` at each step.

**Exercise hooks:**
- *Easy:* Create a repo, make three commits, view the log graph.
- *Medium:* Create a merge conflict on purpose, resolve it, confirm clean merge.
- *Hard:* Rebase a feature branch onto `main` and explain (in a commit message) what changed vs. merge.

## Beat 4: Use It

**GTM redirect: Zone 1 — ICP & Account Intelligence, Zone 2 — Outbound & Enrichment**

GTM teams manage Clay table schemas, enrichment waterfall configs, and scoring models that change weekly. Store these as YAML/JSON in a repo. Branch per campaign, PR for peer review before a new enrichment column goes live. Rollback the scoring model when conversion drops.

**Exercise hooks:**
- *Easy:* Commit a Clay export config (JSON) with a descriptive message.
- *Medium:* Branch for a new ICP experiment, modify the scoring weights, open a PR describing the hypothesis.
- *Hard:* Tag a release (`v1.2-icp-update`) and write a changelog diff between tags.

## Beat 5: Ship It

Set up a shared GTM operations repo with branch protection on `main`, a PR template for campaign changes, and a `README` explaining folder structure. Push two conflicting campaign configs, resolve via PR, merge.

**Exercise hooks:**
- *Medium:* Fork a provided repo, clone locally, create a feature branch with a new outbound sequence config, push, and open a PR.
- *Hard:* Configure branch protection and a `CONTRIBUTING.md` so that no one can push directly to `main` — test that a direct push is rejected.

## Beat 6: 延伸

Git internals: the `.git/objects` store, reflogs for recovering "lost" commits, and how CI/CD triggers map to GTM pipeline stages. Connect to infrastructure-as-code for GTM stacks.

**Exercise hooks:**
- *Medium:* Use `git reflog` to recover a commit you deliberately hard-reset away.
- *Hard:* Write a `pre-commit` hook that validates YAML syntax on any file in `/configs/` before allowing the commit.

---

## Learning Objectives (for `docs/en.md`)

1. **Initialize** a Git repository and explain the working directory → staging → commit pipeline.
2. **Create and merge** branches, resolving merge conflicts with observable output.
3. **Configure** a remote repository and push/pull changes between local and origin.
4. **Implement** a branching strategy (feature branches, PRs, protected `main`) for a GTM operations repo.
5. **Recover** lost commits using reflog and reset — demonstrate with before/after output.

## GTM Redirect Rules Summary

- **Beat 4 (Use It):** Maps to Zone 1 (ICP & Account Intelligence) and Zone 2 (Outbound & Enrichment) — version-controlled configs for enrichment waterfalls, scoring models, and campaign templates stored as code.
- **Beat 5 (Ship It):** Maps to foundational GTM operations practice — the repo structure and collaboration workflow is how GTM engineering teams actually ship and review changes.
- **Beat 6 (延伸):** Foundational for CI/CD pipelines that automate GTM data workflows.
# Lesson Template

Use this template when creating a new lesson. Copy the folder structure and fill in the content.

## Folder Structure

```
NN-lesson-name/
├── code/
│   ├── main.py            (primary implementation)
│   ├── main.ts            (TypeScript version, if applicable)
│   ├── main.rs            (Rust version, if applicable)
│   └── main.jl            (Julia version, if applicable)
├── notebook/
│   └── lesson.ipynb       (Jupyter notebook for experimentation)
├── docs/
│   └── en.md              (lesson documentation)
└── outputs/
    ├── prompt-*.md         (prompts produced by this lesson)
    └── skill-*.md          (skills produced by this lesson)
```

## Documentation Format (docs/en.md)

```markdown
# [Lesson Title]

> [One-line motto — the core idea that sticks]

> **Platform:** This course runs in Claude Code Desktop. Exercises happen in your terminal, not the browser.

**Type:** Build | Learn
**Languages:** Python, TypeScript, Rust, Julia (list what's used)
**Prerequisites:** [List prior lessons needed]
**Time:** ~[estimated time] minutes

## The Problem

[2-3 paragraphs. What can't you do without this? Why should you care?
Make it concrete — show a scenario where not knowing this hurts.]

## The Concept

[Explain with diagrams and intuition. No code yet.
Every Concept beat gets one illustration — tier selected by illustration-pipeline.md logic:
- Sequence/flowchart/decision tree → Mermaid (write inline)
- System diagram/entities/data flow → Excalidraw (generated at Stage 06)
- Abstract concept/metaphor → GLM-image (generated at Stage 06)
Build the mental model before implementation.]

## Build It

> **Open your terminal — this section runs in Claude Code, not the browser.**

[Step-by-step implementation from scratch.
Start with the simplest version, then add complexity.
Every code block should be runnable on its own.]

### Step 1: [Name]

[Explanation]

    [code block]

### Step 2: [Name]

[Explanation]

    [code block]

[...continue...]

## Use It

[Now show how frameworks/libraries do the same thing.
Compare your from-scratch version to the library version.
This proves the concept and introduces practical tools.]

## Ship It

[What reusable artifact does this lesson produce?
Could be a prompt, a skill, an agent, an MCP server, or a tool.
Include it here and save it in the outputs/ folder.]

## Exercises

1. [Easy — reinforce the core concept]
2. [Medium — apply it to a different problem]
3. [Hard — extend or combine with prior lessons]

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| [term] | [common misconception] | [actual definition] |

## Sources

<!-- GTM strand citations — written by Stage 02 during lesson injection -->
<!-- Format: [Source name](url) — [what it supports in this lesson] -->
<!-- Further reading is not surfaced here — Helix serves recommendations on demand from the antilibrary -->
```

## Code File Guidelines

- Code must run without errors
- No comments — code should be self-explanatory
- Use the language that fits best for the topic
- Include a `requirements.txt` or equivalent if there are dependencies
- Start simple, build up complexity
- Every function and class should have a clear purpose

## Output File Format

### Prompts

```markdown
---
name: prompt-name
description: What this prompt does
phase: [phase number]
lesson: [lesson number]
---

[Prompt content]
```

### Skills

```markdown
---
name: skill-name
description: What this skill teaches
version: 1.0.0
phase: [phase number]
lesson: [lesson number]
tags: [relevant, tags]
---

[Skill content]
```

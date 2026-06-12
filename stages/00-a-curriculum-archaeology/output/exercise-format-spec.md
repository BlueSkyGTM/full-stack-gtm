# Exercise Format Spec
<!-- Derived from: LESSON_TEMPLATE.md + phases/01-math-foundations/01-linear-algebra-intuition/docs/en.md -->
<!-- Capture date: 2026-06-12 | Git hash: 56e1283 -->

## Exercise location

Exercises live **inside the lesson doc** (`docs/en.md`) under the `## Exercises` section. There is no separate exercise file per lesson. The copy-paste flag format is locked in 00-d — do not define it here.

## Current format (from repo)

Exercises are a numbered Markdown list at the end of the lesson doc, after Key Terms:

```markdown
## Exercises

1. [Easy task — reinforce the core concept]
2. [Medium task — apply to a different problem]
3. [Hard task — extend or combine with prior lessons]
```

**Concrete example (linear algebra lesson):**
```markdown
## Exercises

1. Implement `Vector.angle_between(other)` that returns the angle in degrees between two vectors
2. Create a 2D scaling matrix that doubles the x-coordinate and triples the y-coordinate, then apply it to [1, 1]
3. Given 5 random word-like vectors (dimension 50), find the two most similar using cosine similarity
4. Verify that the Gram-Schmidt output is truly orthonormal: check every pair has dot product 0 and every vector has magnitude 1
5. Create a 3x3 matrix with rank 2. Verify using rank(). Then explain what geometric object the columns span.
6. Project [1, 2, 3] onto [1, 1, 1]. What does the result represent geometrically?
```

## Difficulty tiers

| Tier | Description | Verb examples |
|------|-------------|---------------|
| Easy (1-2) | Reinforce the core concept from the lesson. Student has the scaffold, just extends it. | Implement, Add, Verify, Create |
| Medium (3-4) | Apply concept to a different problem. Requires transferring the idea. | Apply, Compare, Find, Build |
| Hard (5-6) | Extend or combine with prior lessons. No scaffold. May reference prior phases. | Design, Prove, Combine, Extend |

## Rules

1. **No scaffolded code** — exercises are fully open-ended tasks, not fill-in-the-blank
2. **All exercises are terminal-executable** — they should produce observable output the student can check
3. **Hard exercises may name a prior lesson** — "Using the technique from Phase 03 Lesson 04..."
4. **Minimum 3 exercises, maximum 6** — most lessons land at 4-5
5. **Exercises validate against Learning Objectives** — every objective must have ≥1 exercise covering it

## Exercise verification

There is no copy-paste flag. Exercises produce **artifacts** — files in the student's mission command repo (`signals/examples/`, `handlers/`, `outputs/`, etc.). Helix reads the filesystem on next interaction and detects artifact existence. The artifact is the save point.

Exercises must specify what artifact they produce and where it lands. If an exercise doesn't produce a persistent artifact, it is purely conceptual — Helix cannot gate on it and should not try to.

## Exercise outputs folder

Some exercises explicitly produce artifacts stored in `outputs/`:

```
outputs/
├── prompt-*.md     (AI prompts the student creates)
└── skill-*.md      (skills the student builds)
```

Format for prompt outputs:
```markdown
---
name: prompt-name
description: What this prompt does
phase: [number]
lesson: [number]
---

[Prompt content]
```

Format for skill outputs:
```markdown
---
name: skill-name
description: What this skill teaches
version: 1.0.0
phase: [number]
lesson: [number]
tags: [comma, separated]
---

[Skill content]
```

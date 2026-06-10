---
description: >-
  Before quiz.json or new lesson assessment work, follow lesson-planning skill.
  Gate for Cline, Claude Code, and curriculum edits — not chat-mode tutoring.
alwaysApply: true
---

# Lesson planning gate

**Skill (full bar):** `.claude/skills/lesson-planning/SKILL.md` — invoke `/lesson-planning` or read that file before acting.

**Batch queue:** `quiz-factory/` — operator also follows `quiz-factory/CLAUDE.md`.

## When this gate applies

You **must** read and follow the lesson-planning skill before:

- Creating or editing any `phases/**/quiz.json`
- Running a `quiz-factory` manifest row (`create_quiz` or `fill_explanations`)
- Advising whether a lesson or quiz is acceptable (Consultant/Otto)
- Planning **new** lesson content that will ship `docs/en.md`, `code/`, or quiz

## When it does not apply

- Office-hours / curriculum chat without touching quiz files (`curriculum-chat.md`)
- `/check-understanding` phase exams (separate skill)
- Read-only grep, audit commands, or reviewing without a quality verdict
- Editing `docs/en.md` or `code/` alone — but **new** lessons: objectives in doc **before** quiz (see skill)

## Accountability (non-negotiable order)

1. `docs/en.md` objectives exist and match what you will test.
2. Quiz claims come from **that** lesson’s doc + code — not generic ML trivia or copied flagship wording.
3. `python scripts/audit_lessons.py` passes after the edit (Tier B before fork-complete).
4. One commit per lesson directory for quiz work (`AGENTS.md`).

If you cannot defend a question against `docs/en.md`, mark manifest `blocked` or stop — do not ship a filler quiz.

## Rule vs skill

- **This rule** = when to open the teacher manual (short gate).
- **lesson-planning skill** = how to plan, compare, and review (seven insights, triage).
- Do not duplicate the skill into other rules or `consultant-charter.md`.

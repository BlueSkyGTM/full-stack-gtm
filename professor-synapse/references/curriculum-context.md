# Curriculum Context — AI Engineering From Scratch

This file orients Professor Synapse to the structure of the AI Engineering From Scratch
curriculum. Read this every session before any teaching turn.

---

## What This Curriculum Is

AI Engineering From Scratch (AIFS) is a 498-lesson, 20-phase curriculum that builds
AI engineering skill from raw mathematics through production deployment. Every algorithm
is implemented from scratch before any framework is touched.

This site and this Synapse installation are built on top of the upstream open-source
curriculum by rohitg00. The upstream repo lives at:
  https://github.com/rohitg00/ai-engineering-from-scratch

The student's local copy may include additional phases, patches, or quiz data that
differ slightly from upstream. Use the local phases/ directory as the authoritative source.

---

## File Layout (relative to this repo root)

```
phases/                          ← all lesson content
  00-setup-and-tooling/
  01-math-foundations/
  02-python-for-ml/
  ...
  19-capstone-projects/
    NN-lesson-slug/              ← each lesson directory
      docs/en.md                 ← the primary lesson document (read this for content)
      code/                      ← implementation files
      quiz.json                  ← 5–7 questions (see schema below)
      outputs/                   ← expected outputs / reference solutions

progress/
  aifs-progress.json             ← current lesson, completed lessons, quiz scores (read-only)
  learning-profile.json          ← student preferences, strengths, doubts (read-only)
  synapse-observations.jsonl     ← Synapse's session observations (WRITE HERE — not above)

graphify-out/
  graph.json                     ← concept dependency graph (use scripts/query_graph.py)

scripts/
  query_graph.py                 ← graph navigation (use this before exploring phase files)

.claude/skills/professor-synapse/
  agents/INDEX.md                ← faculty agent registry
  agents/phase-NN-faculty.md     ← phase specialists (loaded when they exist)
  agents/patterns/               ← JSONL observation files per phase
  references/                    ← this directory (reference files Synapse loads)
```

---

## Phase List

| # | Slug | Domain |
|---|------|--------|
| 00 | setup-and-tooling | Python environment, Jupyter, Git |
| 01 | math-foundations | Linear algebra, calculus, probability |
| 02 | python-for-ml | NumPy, Pandas, Matplotlib |
| 03 | classical-ml | Regression, trees, SVMs, clustering |
| 04 | computer-vision | CNNs, object detection, segmentation |
| 05 | nlp-fundamentals | Text processing, embeddings, RNNs |
| 06 | deep-learning | Backprop, optimizers, regularization |
| 07 | transformers | Self-attention, BERT, GPT architecture |
| 08 | llm-fundamentals | Pretraining, tokenization, scaling laws |
| 09 | fine-tuning | LoRA, RLHF, instruction tuning |
| 10 | rag | Retrieval, chunking, vector stores |
| 11 | llm-engineering | Prompting, evals, structured output |
| 12 | ml-ops | CI/CD for ML, monitoring, drift |
| 13 | deployment | Serving, optimization, edge inference |
| 14 | agent-engineering | Tool use, memory, multi-agent systems |
| 15 | ai-safety | Alignment, red-teaming, jailbreaks |
| 16 | multi-agent | Coordination, debate, society of mind |
| 17 | infrastructure | Distributed training, cloud, costs |
| 18 | production-llms | Latency, caching, evaluation at scale |
| 19 | capstone-projects | End-to-end builds across all domains |

---

## Quiz Schema

Each `quiz.json` contains 5–7 questions in this shape:

```json
{
  "questions": [
    {
      "id": "q1",
      "question": "string",
      "type": "multiple_choice | true_false | short_answer",
      "options": ["A", "B", "C", "D"],
      "answer": "A",
      "explanation": "string"
    }
  ]
}
```

Use quiz questions to probe understanding after a lesson discussion. Do not just read
them aloud — use them to find where the student's mental model breaks down.

---

## Graph Navigation

Always prefer the graph for cross-phase connections:

```bash
python3 scripts/query_graph.py query "attention mechanism"
python3 scripts/query_graph.py path "backpropagation" "transformer training"
python3 scripts/query_graph.py explain "layer normalization"
```

If graphify-out/graph.json doesn't exist, fall back to BATCH.md exploration.
Never block on graph availability.

---

## Progress File Schema

`progress/aifs-progress.json`:
```json
{
  "lessons": {
    "phaseId:lessonIndex": "completed | in_progress"
  },
  "focus": {
    "currentLesson": "phaseId:lessonIndex | null",
    "mode": "string | null",
    "note": "string"
  }
}
```

Lesson identity key: `phaseId + ':' + lessonIndex` (e.g. `"07:3"`).

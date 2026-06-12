# Quiz Format Spec
<!-- Derived from: phases/01-math-foundations/01-linear-algebra-intuition/quiz.json -->
<!-- Capture date: 2026-06-12 | Git hash: 56e1283 -->

## File location

```
phases/[NN]-[zone-name]/[NN]-[lesson-name]/quiz.json
```

One quiz.json per lesson. No separate exercise files — exercises live in the lesson doc (docs/en.md).

## JSON Schema

```json
{
  "questions": [
    {
      "stage": "pre" | "check" | "post",
      "question": "Question text",
      "options": [
        "Option A",
        "Option B",
        "Option C",
        "Option D"
      ],
      "correct": 0,
      "explanation": "Why the correct answer is correct, and why the concept matters."
    }
  ]
}
```

### Field definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `stage` | string enum | yes | When this question fires in the lesson flow |
| `question` | string | yes | The question text — complete sentence ending in `?` |
| `options` | string[4] | yes | Exactly 4 options — always 4, never 3 or 5 |
| `correct` | number | yes | Zero-indexed integer (0 = first option) |
| `explanation` | string | yes | Explanation of the correct answer. Required — Helix surfaces this after answer. |

### Stage values and question distribution

| Stage | Count | When it fires | Purpose |
|-------|-------|---------------|---------|
| `"pre"` | 1 | Before reading the lesson | Activate prior knowledge, surface misconceptions |
| `"check"` | 3 | During the lesson (interspersed) | Reinforce each sub-concept as it's introduced |
| `"post"` | 2 | After completing the lesson | Confirm understanding of the full lesson |

**Total per lesson: 6 questions** (1 pre + 3 check + 2 post).

### Concrete example (from phases/01/01)

```json
{
  "questions": [
    {
      "stage": "pre",
      "question": "What does the dot product of two vectors measure?",
      "options": [
        "The distance between the two vectors",
        "How similar or aligned the two vectors are",
        "The angle between the vectors in degrees",
        "The number of dimensions the vectors share"
      ],
      "correct": 1,
      "explanation": "The dot product measures alignment: positive means same direction (similar), zero means perpendicular (unrelated), negative means opposite direction (dissimilar). This is the basis of similarity search in AI."
    },
    {
      "stage": "post",
      "question": "How does LoRA use linear algebra to efficiently fine-tune large language models?",
      "options": [
        "It removes unused rows from weight matrices to shrink the model",
        "It converts all weights from float32 to int8",
        "It decomposes weight updates into two small low-rank matrices instead of updating the full weight matrix",
        "It freezes the embedding layer and only trains the output head"
      ],
      "correct": 2,
      "explanation": "LoRA decomposes a 4096x4096 weight update into two matrices of size 4096x16 and 16x4096 (rank-16), reducing trainable parameters from 16M to 131K by assuming updates live in a low-dimensional subspace."
    }
  ]
}
```

## Question quality rules

1. **Ground every question in the lesson doc** — if you can't cite the specific section it comes from, don't write it
2. **Pre-quiz tests prior knowledge or reveals misconception** — it's OK if the student gets it wrong
3. **Check questions test one sub-concept each** — map 1:1 to Beat 2 sub-sections
4. **Post questions test application, not recall** — "what does this do in AI?" not "what is the definition of..."
5. **Distractors must be plausible** — wrong options should be things a smart person might believe, not obviously silly
6. **Explanation is mandatory** — it teaches on wrong answers; minimum 2 sentences

## Scoring model (for Stage 05 Helix integration)

Helix reads `stage` to gate lesson progression:
- **pre**: no gate — student can proceed regardless of answer
- **check**: soft gate — wrong answer triggers hint, student can continue
- **post**: hard gate — both must be correct to mark lesson complete

FSRS scheduling uses question-level responses (not lesson-level). Each `question.id` (added at Stage 04) maps to an FSRS card. Schema extension for FSRS at Stage 04:

```json
{
  "id": "q1",
  "stage": "post",
  "question": "...",
  "options": [...],
  "correct": 2,
  "explanation": "...",
  "fsrs": {
    "due": null,
    "stability": 0,
    "difficulty": 5,
    "elapsed_days": 0,
    "scheduled_days": 0,
    "reps": 0,
    "lapses": 0,
    "state": 0,
    "last_review": null
  }
}
```

The `fsrs` block is empty at creation — Helix populates it on first answer.

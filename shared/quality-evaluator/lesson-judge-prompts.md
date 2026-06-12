# Lesson Judge Prompts

LLM-as-judge prompts for curriculum quality evaluation. Architecture adapted from learning-commons-org/evaluators. Hypatia runs all judges at Stage 09.

Each prompt takes a lesson file as input and returns a structured score + specific failure notes.

---

## CLARITY_JUDGE

**Purpose:** Score reading clarity and cognitive load per the CLEAR text complexity framework.

```
You are a curriculum quality evaluator. Score the following lesson for reading clarity.

LESSON:
{{LESSON_CONTENT}}

Evaluate against these criteria:

1. SENTENCE_COMPLEXITY (1-4)
   4 = All sentences < 25 words average, none > 40 words
   3 = Most sentences within limit, 1-2 outliers
   2 = Frequent long sentences that require re-reading
   1 = Dense, multi-clause sentences throughout

2. VOCABULARY_LOAD (1-4)
   4 = Max 2 new terms per Beat, each defined on first use
   3 = Terms introduced but definitions delayed or missing
   2 = Multiple undefined terms per Beat
   1 = Assumes vocabulary the target reader cannot have

3. COGNITIVE_DEMAND (1-4)
   4 = One primary concept per Beat, compound ideas get their own Beat
   3 = Occasionally mixes concepts within a Beat
   2 = Beats regularly try to teach 2-3 concepts
   1 = No clear concept-per-Beat structure

4. COHERENCE (1-4)
   4 = Every paragraph leads with a claim; detail follows
   3 = Most paragraphs structured correctly; some meander
   2 = Paragraphs frequently bury the main point
   1 = No consistent paragraph structure

Return JSON:
{
  "lesson_id": "{{LESSON_ID}}",
  "scores": {
    "sentence_complexity": <1-4>,
    "vocabulary_load": <1-4>,
    "cognitive_demand": <1-4>,
    "coherence": <1-4>
  },
  "overall": <average, 1 decimal>,
  "failures": ["<specific line or Beat that fails, quoted>"],
  "pass": <true if overall >= 3.0, false otherwise>
}
```

---

## WEAVE_JUDGE

**Purpose:** Verify GTM strand is woven into AI lesson, not running in parallel.

```
You are a curriculum quality evaluator. Score the GTM-AI weave quality of the following lesson.

LESSON:
{{LESSON_CONTENT}}

Evaluate against these criteria:

1. DERIVATION (1-4)
   4 = GTM section opens with: "Because [AI concept], you can [GTM action]" — mechanical derivation explicit
   3 = GTM section references the AI concept but derivation is implicit
   2 = GTM section is adjacent to AI content but could stand alone
   1 = GTM section makes no reference to the AI concept in this lesson

2. SPECIFICITY (1-4)
   4 = GTM application names a specific tool, workflow step, or metric from the GTM handbook
   3 = GTM application is specific to GTM but not grounded in handbook toolstack
   2 = GTM application is generic ("you can automate this in your GTM stack")
   1 = GTM application is vague or motivational rather than operational

3. SOURCES_PRESENT (1 or 4)
   4 = ## Sources block exists with at least one citation per GTM claim
   1 = ## Sources block is empty or missing

Return JSON:
{
  "lesson_id": "{{LESSON_ID}}",
  "scores": {
    "derivation": <1-4>,
    "specificity": <1-4>,
    "sources_present": <1 or 4>
  },
  "overall": <average, 1 decimal>,
  "failures": ["<specific GTM section that fails, quoted>"],
  "pass": <true if overall >= 3.0 AND sources_present == 4, false otherwise>
}
```

---

## ACCURACY_JUDGE

**Purpose:** Verify AI engineering content accuracy against Made With ML source.

```
You are a curriculum quality evaluator. Score the AI engineering content accuracy of the following lesson.

LESSON:
{{LESSON_CONTENT}}

SOURCE_MATERIAL:
{{SOURCE_LESSON_URL}}

Evaluate against these criteria:

1. CONCEPT_FIDELITY (1-4)
   4 = Core AI concept matches source exactly; no introduced misconceptions
   3 = Core concept correct; minor imprecisions in explanation
   2 = Conceptual simplification that could mislead
   1 = Incorrect or contradicts source

2. CODE_CORRECTNESS (1-4) — skip if lesson has no code
   4 = All code blocks run without errors; output matches description
   3 = Code correct but output description slightly off
   2 = Code has non-fatal errors or deprecated patterns
   1 = Code does not run

3. TERM_PRECISION (1-4)
   4 = Technical terms used exactly as defined in source
   3 = Minor informal usage of 1-2 terms
   2 = Terms used loosely throughout
   1 = Terms used incorrectly

Return JSON:
{
  "lesson_id": "{{LESSON_ID}}",
  "scores": {
    "concept_fidelity": <1-4>,
    "code_correctness": <1-4 or "N/A">,
    "term_precision": <1-4>
  },
  "overall": <average of applicable scores, 1 decimal>,
  "failures": ["<specific claim or code block that fails, quoted>"],
  "pass": <true if overall >= 3.5, false otherwise>
}
```

---

## Usage by Hypatia (Stage 09)

Run all three judges per lesson. A lesson fails Stage 09 if:
- CLARITY overall < 3.0
- WEAVE overall < 3.0 OR sources_present == 1
- ACCURACY overall < 3.5

Failed lessons are flagged as `blocked` in the manifest and routed to Lyra for revision before Stage 10.

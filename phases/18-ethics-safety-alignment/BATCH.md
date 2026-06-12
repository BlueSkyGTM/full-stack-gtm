# Phase 18 — Ethics, Safety, and Alignment (quiz factory)

## Focus

AI safety and alignment: RLHF, reward hacking, Goodhart's Law, scalable oversight, interpretability, red-teaming, jailbreaks, watermarking, dual-use risks, and constitutional AI / self-critique methods.

## Scrape hints

- `docs/en.md`: definitions of alignment failure modes, threat model diagrams, empirical results from the lesson's safety evaluations, policy implications
- `code/main.py`: often demonstrates a reward model, a toy red-teaming loop, or a safety classifier — quiz what the code tests and what it can/cannot catch
- Vocabulary: `glossary/terms.md` for reward hacking, Goodhart's Law, scalable oversight, deceptive alignment, jailbreak, watermark, constitutional AI

## Phase 18 batch note

All Phase 18 rows in the manifest are `fill_explanations` jobs — quiz files exist with valid questions but empty `explanation` fields. Write 1–3 sentence explanations from `docs/en.md`. Do not change question text, options, or `correct` index.

## Style anchor

- No gold Phase 18 quiz produced yet — use the flagship lesson-planning SKILL.md guidelines for depth
- Explanations on safety topics must cite the **lesson's** empirical claims (e.g., specific paper's results), not generic safety platitudes

## Common explanation pitfalls

- Generic claims ("AI can be unsafe") instead of the lesson's specific mechanism
- Restating the correct option verbatim with no added pedagogical value
- Importing claims from other safety papers not cited in the lesson's doc

## Do not

- Change any question text, options, or `correct` index during `fill_explanations`.
- Import facts from other phases unless `docs/en.md` lists them as prerequisites.
- Ask the user questions — mark `blocked` in manifest instead.

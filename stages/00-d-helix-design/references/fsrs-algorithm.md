# FSRS Algorithm Reference

Source: https://github.com/open-spaced-repetition/awesome-fsrs/wiki/The-Algorithm

## What to Read

Before running 00-d, read the algorithm spec and test assets at the URL above. Key sections:

- Card states (New, Learning, Review, Relearning)
- Scheduling parameters (w values)
- Stability and difficulty calculations
- Correct-response definition for text-based recall

## Integration Notes

This is a text-based curriculum with CLI exercises. "Correct response" must be defined behaviorally (student submits copy-paste flag) rather than via a graded answer. Document this decision in fsrs-integration-spec.md.

Test assets exist — use them to validate the implementation in Stage 05.

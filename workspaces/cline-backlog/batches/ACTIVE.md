# ACTIVE — Site Integration

Schema repair is complete (all 351 quizzes canonical as of commit cdea0b0, 2026-06-03).

Two site integration items remain:

## 1. Quiz UI in lesson reader
Questions exist as data in each phase's `quiz.json`. Nothing renders them in the site.
`site-new/lesson.html` needs a quiz section that loads `quiz.json` for the current lesson and displays it after the lesson content.

## 2. data.js sync — Phase 19 lessons 58–87
30 new capstone lessons exist in `phases/19-capstone-projects/` (58–87) but are not yet
indexed in `site-new/js/data.js`. The file is generated — to update it, re-run the
curriculum build or manually add the 30 entries.

---

## Done (archived 2026-06-04)

- All schema repairs — 5q → 6q, 7q → 6q, 8q → 6q across phases 05, 14, 19, and others.
  128 repairs in cdea0b0, 132 in 6636c03. All 351 quizzes canonical.
- Phase 19 lessons 58–87 pulled from upstream (30 files, done in d113ede).
- Quiz content created from scratch — eliminated. 200+ quiz files exist in upstream.

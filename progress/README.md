# Learner progress file

`aifs-progress.json` in this directory is the **repo copy** of site progress (optional, for Mac/PC sync and for Cursor agents).

## How it gets data

1. Learn on [aiengineeringfromscratch.com](https://aiengineeringfromscratch.com) (visits, quizzes, mark complete).
2. **Save** in the site header → download `aifs-progress.json`.
3. Copy or replace this file and `git commit` if you want agents and other machines to see the same state.

## Schema

```json
{
  "lessons": {
    "phases/07-transformers-deep-dive/12-kv-cache-flash-attention": {
      "visitedAt": 1717000000000,
      "completedAt": null,
      "answers": {
        "pre-q0": { "picked": 1, "correct": true, "t": 1717000000000 }
      }
    }
  },
  "updatedAt": 1717000000000,
  "savedAt": 1717000000000,
  "message": "optional save label",
  "focus": {
    "currentLesson": "phases/07-transformers-deep-dive/15-attention-variants",
    "mode": "office-hours",
    "note": "optional free-text for the agent"
  }
}
```

- `focus` is optional. Agents read it per `.cursor/rules/curriculum-chat.mdc`. The site does not write `focus` yet; you can edit it by hand or ask the agent to remember the current lesson in chat.

## Learning profile (`learning-profile.json`)

Optional. Created via **`/learning-style-setup`** in Cursor. Agents read it for **guidance counselor** mode; they do not write it.

Fields: `preferences` (pace, depth, feedback, session length), `strengths`, `weaknesses`, `doubts`, `avoid`, `notes`, `setupComplete`.

Set `focus.mode` to `guidance` when you want curriculum-chat to treat the thread as meta (optional):

```json
"focus": {
  "mode": "guidance",
  "note": "doubting phase 7 pace"
}
```

## Privacy

Quiz and completion data only — no API keys. Public commit = public learning log (FCC-style); use a private fork if you prefer. `learning-profile.json` may include personal doubts — keep repo private if that matters.
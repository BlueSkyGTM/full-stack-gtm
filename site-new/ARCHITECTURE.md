# Architecture & Expansion Guide

How content flows through AI School, and **exactly what to touch (and what to leave alone)** when
you grow the curriculum. Read the two "hazard" sections before adding anything — a couple of edits
look harmless but silently corrupt saved progress or leave parts of the app stuck at "20 worlds."

---

## 1. Data flow

```
ai-engineering-from-scratch/   (curriculum source repo: markdown lessons)
            │
            ▼  build.js   ← runs in the source repo, NOT here
   js/data.js   ── generated bundle: defines PHASES + GLOSSARY (globals)
            │
            ▼  loaded first on every page
   store.js (progress) ──▶ game.js (pure derive) ──▶ screen.js (render)
```

- **`js/data.js` is a build artifact.** Header says so. Never hand-edit it — your change is lost on
  the next build. Edit the source repo's markdown, re-run *its* `build.js`, copy the new `data.js` in.
- `PHASES` and `GLOSSARY` are plain globals. No imports, no fetch. Adding a page = add an HTML file
  that loads `data.js` first, then your screen script.

---

## 2. ⚠️ Hazard #1 — lesson keys are POSITIONAL

Progress is stored under the key **`phaseId + ':' + lessonIndex`** (see `store.js → key()`).
`phaseId` is an explicit, stable field. **`lessonIndex` is the array position.** Consequences:

| Action | Safe? | Why |
|---|---|---|
| **Append** a lesson to the end of a phase | ✅ safe | existing indices unchanged |
| **Add** a whole new phase (new `id`) | ✅ safe | new keys, no collision |
| **Insert / reorder** lessons inside a phase | ❌ **breaks saved progress** | every later index shifts → a learner's "done" ticks land on the wrong lessons |
| **Renumber / reorder** phase `id`s | ❌ breaks | same problem at the phase level |

**If you must insert or reorder**, you are changing the key space — bump the storage version and
migrate:
1. In `store.js`, change `KEY = 'aischool.progress.v1'` → `…v2`.
2. Add a one-time migration that remaps old keys to new positions (or accept a clean reset for the
   demo). Without this, learners silently lose/scramble progress.

**Best practice for routine growth:** only ever *append* lessons within a phase, and *append* new
phases with fresh ids. That path needs zero store changes.

---

## 3. ⚠️ Hazard #2 — hardcoded "20 phases / boss = 19"

Several places assume the current shape of the curriculum. When you expand, grep for the phase id
and update each. **This is the full list — there are no others:**

| What | File | Current assumption | Update when… |
|---|---|---|---|
| Prerequisite graph `PREREQS` | `js/hub.js` **and** `js/roadmap.js` | keys `0–19` (duplicated in both) | every new phase needs an entry in **both** copies (drawer chips + dependency graph) |
| World→land grouping `LANDS` | `js/hub.js` | 6 lands covering phases `0–19` | every new phase must be added to a land (or a new land) or it won't appear on the Home map |
| Boss node | `js/hub.js` (`pin--boss`), `js/roadmap.js` (`tnode--boss`) | `p.id === 19` | you add phases *after* 19, or move the capstone phase |
| Boss/capstone list | `js/projects.js` | `PHASES.find(p => p.id === 19)` | same as above |
| Badge rules `BADGES` | `js/game.js` | reference phases `0,1,2,3,14` | you want badges for new phases (else new content earns none) |
| Rank ladder `RANKS` | `js/game.js` | tops out at level 19 | more lessons → more XP → learners may exceed the top rank; extend the ladder |
| Core languages `CORE_LANGS` | `js/game.js` | 4 fixed langs | a new phase introduces a language you want counted |
| Demo seed | `js/store.js → seedIfEmpty` | hardcodes phases `0,1,2` | only if you want the first-open demo state to reflect new phases |
| Headline counts | `index.html` | literal `"twenty worlds"` in the subtitle | every expansion that changes the world count |
| Capstone-grid cap | `js/projects.js` | `CAP = 18` shown, rest → "+N more" | only if Phase 19's lesson count changes a lot and you want more/fewer featured |

Everything else scales automatically: the Home map re-flows its pins, Course/Roadmap/Glossary all
iterate `PHASES`/`GLOSSARY`, and every XP total and percentage derives from the data.

---

## 3a. Tip: kill the stale-count problem for good

`index.html`'s subtitle is the one number that rots on every expansion. Optional one-time fix so it
never lies again — derive it at render time instead of hardcoding:

```js
// in hub.js, after stats are derived:
$('.mast__sub').textContent =
  `${PHASES.length} worlds · ${stats.lessonsTotal} lessons · build every algorithm from scratch, one world at a time.`;
```

---

## 4. Expansion playbooks

**Add a lesson (safe path):** append a lesson object to the end of a phase's `lessons[]` in the
*source repo*, rebuild, drop in the new `data.js`. Done — XP, totals, catalog, TOC all update. No
app-code changes.

**Add a phase:** give it the next free `id`, append to `PHASES` (source → rebuild). Then update the
coupling rows that apply: add a `PREREQS[newId]` entry in **both** `hub.js` and `roadmap.js`; drop
the phase id into a `LANDS` entry in `hub.js` (or add a new land); if it's a new boss/capstone, move
the `=== 19` checks; add any badge you want (game.js); refresh the headline subtitle.

**Add a glossary term:** append to `GLOSSARY` in source, rebuild. Glossary page is fully
data-driven — no other changes.

---

## 5. Resource discipline (don't bloat the build)

`js/data.js` is already ~640 KB and **every page loads all of it**, including pages that only need
part (glossary needs `GLOSSARY`, not `PHASES`). It grows linearly with the curriculum. To expand
without wasting payload/review effort:

- **Keep generated fields lean.** `summary` and `keywords` are the bulk. Cap their length in
  `build.js`; don't dump full lesson bodies into `data.js`.
- **Split the bundle when it gets big.** Generate `phases.js` and `glossary.js` separately so each
  page loads only what it renders (e.g. Glossary needs only `GLOSSARY`; Projects needs only Phase 19).
  The globals contract stays the same; only the `<script>` tags change.
- **Don't rebuild blind.** A content typo is a source edit + targeted rebuild, not a reason to
  regenerate and re-review the whole 12k-line file by hand. Diff `data.js` before committing.
- **Cache-bust on data change only.** If you add a query string / hash to the `data.js` script tag,
  bump it when data changes — not on every deploy — so the big file stays cached.

---

## 6. What NOT to touch

- `js/data.js` by hand (regenerated).
- The `phaseId:index` key format without a version bump + migration (§2).
- `css/tokens.css` variable *names* — components reference them everywhere; change values, not names.
- `screenshots/`, `uploads/`, `AI School Wireframes.html` — not part of the running app.

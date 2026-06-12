# AI School

A gamified world-map front-end for the **AI Engineering From Scratch** curriculum — 20 "worlds"
(phases), 473 lessons, XP / levels / ranks / badges / streaks, all tracked locally.

Open `index.html` in a browser. No build, no server, no dependencies (fonts load from Google
Fonts; the lesson reader fetches lesson markdown from GitHub at runtime). Progress saves to
localStorage today and is built to move to **WordPress** (swap `Store.adapter` in `store.js`).

---

## How it fits together

Vanilla JS, no framework. Everything hangs off a single global, `window.AIS`. Each page loads its
scripts in a fixed order — **do not reorder**:

```
data.js  →  store.js  →  game.js  →  ui.js  →  <screen>.js
```

- **`js/data.js`** — *generated, do not hand-edit.* Defines `PHASES` and `GLOSSARY`. Rebuilt from
  the curriculum repo (`ai-engineering-from-scratch/build.js`); to change content, edit the source
  repo, re-run its build, and replace this file.
- **`js/store.js`** (`AIS.store`) — the only code that touches persistence. localStorage today;
  swap `Store.adapter` to point at a WordPress REST backend (one line — see the comment in the file).
- **`js/game.js`** (`AIS.game`) — pure rules. `derive(PHASES, progress)` returns the stats object
  every screen renders. No side effects, no DOM.
- **`js/ui.js`** (`AIS.ui`) — tiny `el()` DOM builder + `toast()`.
- **Screens** — one file per page, render-only:
  - `hub.js` — **Home** orientation: hero + signpost cards + world map (archipelago of 6 lands; owns the `LANDS` grouping + phase drawer)
  - `course.js` — **Course** (player card + curriculum accordion)
  - `roadmap.js` — **Roadmap** (the prerequisite dependency graph only; click-to-select)
  - `catalog.js` — **Catalog** (searchable/sortable lesson index with filter chips)
  - `library.js` (+ `library-data.js`) — **Library** (curated tiered reading list)
  - `lesson.js` — **Lesson reader** (fetches + renders lesson markdown; sidebar, mark-complete, prev/next)
  - `projects.js` — **Projects** (featured work, capstone builds, achievement badges)
  - `glossary.js` — **Glossary** (flip cards, `?q=` deep-links)
  - `cmdpalette.js` — ⌘K command palette (search lessons + glossary), loaded on every page

Styling is fully tokenized in **`css/tokens.css`** — change a variable, the whole app reskins.
`css/components.css` and `css/library.css` reference those vars only.

---

## Art is placeholder slots, not files

There are **no image assets** in this project. Every place real pixel art belongs renders a
`.slot` element (a striped placeholder) with a text label of its intended size. Replace the slot
markup with real `<img>`/sprite when art exists. Current slots:

| Where | File | Label / intended art |
|---|---|---|
| Home HUD avatar | `js/hub.js` | `avatar 16×16` |
| Course player-card avatar | `js/course.js` | `pixel avatar 32×32` |
| Project cover art (featured + capstone builds) | `js/projects.js` | `cover 16×9` / `build 16×9` / `🔒 locked` |
| Achievement badge icons | `js/projects.js` | `★` (earned) / `?` (locked) — one per badge |

When wiring real art: keep the `.slot` class off the new element (it's the placeholder styling),
size to the label, and pull from a single `assets/` folder so discovery stays mechanical.

---

## Not part of the running app

- `screenshots/`, `uploads/` — scratch images, referenced by nothing.
- `AI School Wireframes.html` — a static design-exploration doc.

---

## Data model quick reference

- A lesson is identified by `phaseId:lessonIndex` (e.g. `2:5`). Keep this stable — the store keys on it.
- `PHASES[n]` = `{ id, name, status, desc, lessons:[{ name, status, type, lang, url, summary, keywords }] }`.
- Lesson `type` drives XP: `Learn` 60 · `Build` 120 · `Capstone` 500 (see `game.js`).
- Phase 19 is the "boss" (capstones); the prerequisite graph (`PREREQS`) lives in **both**
  `js/hub.js` (drawer chips) and `js/roadmap.js` (the dependency graph) — edit both if you change it.
- The Home map's phase→land grouping is the `LANDS` array in `js/hub.js`.

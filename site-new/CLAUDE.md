# AI School — project context

Gamified ("world map" RPG) front-end for the **AI Engineering From Scratch** curriculum.
Vanilla JS + CSS, **no build step for the app itself, no framework**. Designed to drop into
WordPress later (see the store adapter note below).

## Golden rules
1. **`js/data.js` is generated — never hand-edit it.** It is built from the curriculum repo
   (`ai-engineering-from-scratch/`, via its `build.js`). It defines two globals: `PHASES`
   (the 20 worlds + their lessons) and `GLOSSARY` (term cards). To change content, change the
   source repo and re-run its build, then replace `js/data.js`.
2. **All imagery is placeholder slots, not files.** There are *no* binary art assets wired in.
   Anywhere real pixel art goes, the code renders a `.slot` element with a text label of the
   intended size (e.g. `'avatar 16×16'`). See the slot table in `README.md`. Do not go looking
   for missing image files — none are referenced.
3. **Style only through tokens.** Every color/space/font is a CSS var in `css/tokens.css`.
   Components reference vars only — change a token, the whole app reskins. Don't hard-code
   colors downstream.
4. **One source of state.** `js/store.js` is the *only* thing that touches persistence
   (localStorage today). Screens never read storage directly — they read derived stats from
   `js/game.js` (a pure function). **This site is being plugged into WordPress**, which will
   own auth + progress persistence: when that happens, swap `Store.adapter` (one line,
   documented in `store.js`) to read/write the WP REST API (e.g. user meta) instead of
   localStorage — no screen code changes. There is deliberately **no Export/Import UI** in the
   app; progress sync is WordPress's job, not a JSON-file dance. (`store.exportJSON()` still
   exists as a helper but nothing in the UI calls it.)

## Architecture (load order matters)
Everything namespaces onto `window.AIS`. Each HTML page loads scripts in this order:
`data.js` → `store.js` → `game.js` → `ui.js` → `<screen>.js`

| Layer | File | Role |
|---|---|---|
| Data | `js/data.js` | **generated.** `PHASES`, `GLOSSARY` globals (each lesson carries `url`/`summary`/`keywords`) |
| Library data | `js/library-data.js` | the `LIBRARY` global (tiered reading list); plain data, hand-maintained |
| Persistence | `js/store.js` | `AIS.store` — read/write/toggle progress; adapter-swappable (→ WordPress) |
| Rules (pure) | `js/game.js` | `AIS.game.derive(PHASES, progress)` → stats; XP, levels, ranks, badges, streaks |
| DOM helper | `js/ui.js` | `AIS.ui.el(...)` builder + `toast()`. No framework |
| Search | `js/cmdpalette.js` | ⌘K / Ctrl-K command palette — client-side search over PHASES + GLOSSARY. Self-contained (no AIS dep); loaded on every page after `data.js` |
| Screens | `js/hub.js` `course.js` `roadmap.js` `catalog.js` `library.js` `projects.js` `glossary.js` `lesson.js` | one file per page; render-only |

Nav order (all pages): **Home · Course · Roadmap · Catalog · Library · Projects · Glossary**,
plus a ⌘K search button in the header (every page links `css/cmdpalette.css` + `js/cmdpalette.js`).

Pages:
- `index.html` — **Home**: orientation. A hero (what this is) + "Where to go" signpost cards
  + the world map. 20 phases grouped into 6 themed *lands* that stack down the page
  (archipelago trail); each phase is a pin on a dotted road. `hub.js` owns the `LANDS` array
  (which phase sits in which land) + the slide-in phase drawer.
- `course.html` — **Course**: the player card (level/rank/XP/career meters) pinned at the
  top, then the curriculum accordion. `course.js`.
- `roadmap.html` — **Roadmap**: the prerequisite dependency **graph only** (the DAG). Click a
  world to select it — lights its upstream prereqs + downstream unlocks, dims the rest;
  "Clear selection" resets. Hover previews. `roadmap.js`.
- `catalog.html` — **Catalog**: the searchable lesson index — big search box + type/status
  filter chips + a sortable manifest of all ~470 lessons. `catalog.js`.
- `library.html` — **Library**: curated free reading list, grouped Fundamentals / Intermediate /
  Advanced with topic filter chips. Data in `js/library-data.js`, render in `js/library.js`.
- `lesson.html` — **Lesson reader**: resolves `?path=phases/<phase>/<lesson>` (or `?p=phaseId:idx`),
  fetches that lesson's markdown from the curriculum repo at runtime, renders it 8-bit with a
  sidebar + mark-complete + prev/next. `js/lesson.js` has a compact markdown→HTML renderer.
  Catalog rows, the Course accordion, the world-map drawer, and ⌘K results all link here.
- `projects.html` — **Projects**: trophy case. Featured-work placeholder slots, the Phase-19
  capstone builds (locked/greyed until cleared), and achievement badges. `projects.js`.
- `glossary.html` — **Glossary**: flip-to-reveal term cards. Supports `?q=Term` deep-links (used by ⌘K). `glossary.js`.

## Ignore these (not app assets)
- `screenshots/`, `uploads/` — working captures/uploads, **not referenced by any code**.
  (`uploads/site/` is a prior blueprint-blue build by Claude Code — the source the lesson reader,
  Library, and ⌘K palette were ported *from*, then re-skinned terracotta. Not part of the running app.)
- `AI School Wireframes.html` — a standalone design exploration doc, not part of the running app.

## Conventions
- IDs in markup are render targets (`#map`, `#drawer`, `#hud`, `#toc`, `#tree`, `#body`,
  `#builds`, `#featured`, `#trophies`, …); screen JS fills them via `replaceChildren`.
- `data-screen-label` marks high-level screens; keep them on the semantic equivalent if you restructure.
- Lesson identity is `phaseId + ':' + lessonIndex` — keep that key stable if you touch the store.
- **`PREREQS` (the dependency graph) is duplicated** in `hub.js` (drawer "recommended after"
  chips) and `roadmap.js` (the dependency graph). Edit both, or hoist it into `data.js` if you
  change it often.
- The world-map phase→land grouping lives in `LANDS` in `hub.js`. Add a new phase to a land
  (or add a land) there — see `ARCHITECTURE.md` §3.
- **Lesson reader markdown** is fetched live from the curriculum repo (raw GitHub) derived from
  each lesson's `url`; it tries `README.md` → `docs/en.md` → `index.md`. WordPress hosting won't
  change this (it's a client-side fetch), but if lessons move repos, update the URL derivation in
  `js/lesson.js` (`loadLesson`).

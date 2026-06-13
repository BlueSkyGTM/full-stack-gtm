# Lyra Code Tailoring Brief
<!-- Stage 00-c output | 2026-06-12 -->
<!-- Agent: /build-site-component → GLM-5 -->

## Identity

Lyra Code is the site engineering agent. It writes, modifies, and extends the full-stack GTM site (site-new/) — component by component, with no breaking changes. It knows the rendering pipeline, the auth flow, the data layer, and the CSS token system by heart. It never guesses at the existing architecture — it reads design-system-snapshot.md and auth-audit.md before touching any file.

## Architecture Invariants (Lyra Code Must Never Violate These)

The following properties of the site are locked. Any change that violates one of them is a breaking change and must not be shipped:

1. **No build step for the app.** The site is vanilla HTML/CSS/JS. No webpack, no bundler, no transpiler. `build.js` is the only build artifact, and it only generates `site-new/js/data.js`.
2. **Script load order is mandatory.** Every page loads: `data.js → store.js → game.js → ui.js → [screen].js`. A new screen file must follow this order.
3. **`data.js` is generated, never hand-edited.** Rebuild by re-running `build.js` in the curriculum repo.
4. **CSS tokens only.** No hard-coded color values or pixel measurements. Every value comes from `site-new/css/tokens.css`. Adding a new value requires a new token.
5. **The site is read-only content delivery.** Progress is NOT tracked by the site. The canonical student state lives in `progress/progress.json` inside the student's mission command fork (the Albatross). The site auth backend (`api/auth.js`, `api/progress.js`, `api/_lib/auth.js`) is gutted in Stage 07. Do not extend it.
6. **No Vercel KV progress writes.** The `vercelAdapter` in `store.js` is removed in Stage 07. If a screen file currently calls `AIS.store.write()`, it writes to nothing (or a no-op stub). Progress lives in the repo, not the server.

## What Lyra Code Reads Before Touching Any File

Required reads at Stage 05+ entry:
- `stages/00-a-curriculum-archaeology/output/design-system-snapshot.md` — rendering stack, CSS tokens, script load order, component patterns
- `stages/00-a-curriculum-archaeology/output/auth-audit.md` — auth flow, adapter pattern, known failure modes, Stage 07 implications
- The specific file it is about to modify — no blind writes

## Component Writing Rules

When writing a new component or screen file:
- Use `AIS.ui.el()` for DOM creation (defined in `ui.js`)
- Use `AIS.store.read()` and `AIS.store.write()` for all progress I/O
- Use `AIS.game.derive()` for XP/level/badge computation — do not reimplement game logic
- Register under `window.AIS` namespace
- Load order: the new screen file is always the last script loaded on its page
- No inline styles — always CSS classes from `tokens.css` or `site-new/css/`

## Helix Integration Rules (Stage 05)

Helix is the student-facing AI tutor (not gstack's gbrain). When implementing Helix components:
- Student identity key: GitHub username from `sa_user` cookie (`login` field)
- FSRS state lives in a separate namespace from progress — it must survive `adapter.clear()` on logout
- Helix prompt injection: read `vault/helix-voice.md` before writing any Helix-facing UI strings
- Helix architecture: read `vault/helix-architecture.md` before touching the Helix component layer

## Auth Gutting Rules (Stage 07)

Stage 07 removes the site auth backend entirely:
- `api/auth.js`, `api/progress.js`, `api/_lib/auth.js` — DELETE these files
- `site-new/js/auth.js` — REMOVE or replace with a stub that does nothing
- `site-new/js/store.js` — REMOVE `vercelAdapter`; keep `localAdapter` only if needed for cosmetic visit log

The site becomes a pure static site. No GitHub OAuth. No Vercel KV. No sa_user cookie.

Do not add guards to the auth API — the auth API does not exist after Stage 07.

## What NOT To Do

- Do not add a build step, bundler, or transpiler
- Do not hard-code color values or pixel measurements
- Do not modify `data.js` by hand — always via `build.js`
- Do not extend or fix the auth backend — it is being removed, not improved
- Do not add Vercel KV progress writes — progress lives in the mission command repo
- Do not add third-party JavaScript libraries without explicit approval — each addition requires a performance audit
- Do not implement Helix behavior in a screen file — Helix logic belongs in its own component
- Do not add site-side gate checks — gates are enforced by Helix reading the student's filesystem, not by the site

## Invocation Pattern

```bash
# Build the Helix student state component per Stage 05 spec
/build-site-component spec="stages/05-helix-build/specs/helix-student-state.md"

# Extend auth flow with env guards per Stage 07 spec
/build-site-component spec="stages/07-student-state/specs/auth-env-guard.md"
```

Triggered by: Stage 05 (Helix build), Stage 06 (site readability), Stage 07 (student state), Stage 08 (agent wiring).

## GLM Model

`GLM-5` — high capability for code generation with site architecture reasoning. Not GLM-5.1 (reserved for content). Never downgrade to GLM-5-Turbo for site component work.

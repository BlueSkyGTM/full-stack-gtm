# Design System — Fleet Engine Portfolio

## Product Context
- **What this is:** Engineering portfolio documenting a 3-agent AI GTM enrichment pipeline (Ahab → Nemo → Neptune)
- **Who it's for:** Hiring managers, technical recruiters, and engineers. Non-programmers should be able to follow the story without getting lost.
- **Space/industry:** AI engineering, GTM tooling, B2B SaaS infrastructure
- **Project type:** Engineering portfolio — three pages (index, v2 Cortex, v1 Fathom)

## Memorable Thing
"A real pipeline that ran 1,500 cycles, broke 18 ways, and got rebuilt until it worked."

## Aesthetic Direction
- **Direction:** Retro-Futuristic / Tech-Gaming (Mega Man Battle Network-inspired)
- **Decoration level:** Intentional — subtle grid patterns, colored section tints, HP bars, battle chip UI. Not ornamental.
- **Mood:** Dark, precise, gamified without being childish. Each section feels like a different area of a network. The portfolio is a tutorial you can play.
- **Anti-slop:** No purple gradients, no 3-column icon grids, no centered everything, no decorative blobs, no gradient buttons.

## Content Structure (Technical Tutorial Workflow)
Each section of v2.html is a step in building the pipeline:
- **Hero**: Frame — "I built this. Here's how it works."
- **Step 01**: The Code — three code layers (Node.js, Python ADK, Agent Prompts)
- **Step 02**: What Breaks — 18 documented V1 failure modes as a troubleshooting reference
- **Step 03**: Run It — one command, three agents, GCS gates
- **Step 04**: Validate — Pydantic output contracts and control layer

## Typography
- **Display/Hero headings**: Cabinet Grotesk 700/800 — geometric, bold, game-UI feel
- **Body**: DM Sans 300/400/500 — already loaded, clean and readable
- **Code/Terminals**: JetBrains Mono — clean monospace, better than Courier New
- **Loading**: Fontshare (Cabinet Grotesk) + Google Fonts (DM Sans + JetBrains Mono)
- **Scale**:
  - Hero h1: clamp(48px, 6.6vw, 92px)
  - Section h2: clamp(38px, 4.4vw, 64px)
  - Lead paragraph: 17-18px
  - Body: 13-14px
  - Labels/captions: 10-12px

## Color

### Agent/Version Colors (canonical)
```
--fe-green:  #B9E36A  — Ahab / V1 Fathom
--fe-blue:   #76A1FF  — Nemo / V2 Cortex
--fe-violet: #9B6DFF  — V3 (locked)
--fe-purple: #D7B7FF  — Neptune (synthesis)
--fe-amber:  #F0A030  — Python layer
--fe-red:    #FF5555  — Failures / errors
```

### Core Palette
```
--bg:       #0A0A0A  — Base
--surface:  #111111  — Card surface
--surface2: #161616  — Secondary surface
--border:   #1E1E1E  — Subtle border
--border2:  #2A2A2A  — Stronger border
--text:     #E6E6E6  — Body text
--muted:    #6A6A6A  — Dimmed text
```

### MMBN Section Area Backgrounds
Each page section has its own distinct tinted background + pattern:
```
§01 languages:     #030810  + rgba(118,161,255,0.08) radial dot grid
§02 failures:      #140306  + rgba(255,85,85,0.05) horizontal scan lines
§03 orchestration: #030B03  + rgba(185,227,106,0.04) vertical pulse lines
§04 python:        #0D0800  + rgba(240,160,48,0.06) circuit crosshatch
```

### Index Panel Backgrounds
```
V1 panel: #050A03  — dark green tint (Fathom era)
V2 panel: #03070F  — dark blue tint (Cortex era)
V3 panel: #08041A  — dark purple tint (locked)
```

## Spacing
- **Base unit**: 8px
- **Density**: Comfortable
- **Scale**: 4 8 16 24 32 48 64 96 104

## Layout
- **Approach**: Grid-disciplined
- **Max content width**: 1080-1200px centered
- **Sections**: Full-bleed section backgrounds, inner content padded 36-104px vertical / 36-48px horizontal
- **Interactive panels** (failure-log, pattern-explorer): detail on LEFT (1.4fr), list on RIGHT (0.6fr)
- **Border radius**: None (sharp corners — MMBN aesthetic)
- **Panel borders**: 2px solid rgba(agent-color, 0.50) + glow shadow

## Motion
- **Approach**: Intentional only — motion signals something happened, doesn't perform
- **Easing**: ease-out entering, cubic-bezier(0.16,1,0.3,1) for HP bars (spring feel)
- **Duration**: reveal 0.7s, HP bar fill 1.2s, terminal pulse 0.6s staggered, hover 0.15-0.2s
- **HP bars**: animate 0→target on scroll enter (IntersectionObserver, threshold 0.5)
- **§03 terminals**: header background pulses once (green/blue/purple, 0/100/200ms stagger) on section enter
- **DO NOT add**: character-by-character scan-in, typing animations, scroll-linked parallax — slows readers

## Gamification Elements
- **HP bars**: 4px colored bar under each stat number, fills according to value
- **Battle chip tabs**: thick bordered filter cards with corner notch, glow on active
- **Section labels**: monospace bordered prefix (NET area marker style)
- **Agent terminals §03**: each terminal column gets its agent's color (green/blue/purple) for header and border

## Page-Level Notes

### index.html
- **Direction**: Orient-first — h1 sets context, tight 2-sentence deck, right column shows crew, panels explain versions
- **h1**: "Fleet Engine" (large, Cabinet Grotesk 800) + subline "A three-agent GTM pipeline"
- **Balance fix**: Left column taller h1 balances right column crew section
- **Panels**: Solid color tint backgrounds only (no dot-grid patterns on panels)

### v2.html
- **Direction**: Technical Tutorial — each section is a numbered step
- **Section labels**: "Step 01", "Step 02", "Step 03", "Step 04"
- **§03 terminals**: Ahab=green, Nemo=blue, Neptune=purple headers + left border per agent color

### v1.html
- **Direction**: MMBN polish only — same fonts, same battle-chip tabs, stronger green section backgrounds. Content unchanged.
- **Palette**: Green-first (--accent: #B9E36A dominates)

## Decisions Log
| Date       | Decision                                 | Rationale |
|------------|------------------------------------------|-----------|
| 2026-06-05 | MMBN aesthetic direction                 | User direction — gamified portfolio |
| 2026-06-05 | Cabinet Grotesk for display headings     | More distinctive than DM Sans, geometric game-UI quality |
| 2026-06-05 | JetBrains Mono for code                  | Cleaner than Courier New, Google Fonts reliable |
| 2026-06-05 | Content → Technical Tutorial Workflow    | Content leads design per user instruction |
| 2026-06-05 | v1 = MMBN polish only, not full rewrite  | Archive page retains green-first character |
| 2026-06-05 | Index = orient-first                     | Panels sell the versions; header just sets context |
| 2026-06-05 | Stronger section tint differentiation    | Previous tints were too subtle — sections looked identical |
| 2026-06-05 | Detail pane on LEFT, list on RIGHT (§02/§04) | Mouse-side list; user confirmed |

# AI Engineering From Scratch

> Build modern AI, by hand.

503 lessons across 20 chapters — ~320 hours of AI engineering from linear algebra through autonomous agent swarms. Python, TypeScript, Rust, Julia. Every lesson follows the same six-beat structure:

```
MOTTO → PROBLEM → CONCEPT → BUILD IT → USE IT → SHIP IT
```

The "Build It / Use It" split is the core idea: you implement the algorithm from raw math first, then run the same thing through PyTorch or scikit-learn. By the time the framework shows up, you already know what it's doing. Over 2,000 quiz questions across 351 lesson files. No hand-holding, no five-minute videos.

---

## Dashboard

Everything in `site-new/` is original. No framework. No build step. Opens directly in a browser.

| Page | What it does |
|------|-------------|
| **Course** | 20-chapter accordion. Per-lesson tick boxes. Live XP bar and rank. |
| **Catalog** | All lessons indexed. Filter by chapter, type, language, or search by name. Rows clickable — opens the lesson reader. |
| **Lesson reader** | Fetches and renders lesson markdown from GitHub. Sidebar shows all lessons in the current chapter with completion state. Prev/next navigation. |
| **Projects** | 17 capstone builds from Phase 19, locked until you complete the prerequisite lessons. |
| **Library** | 110+ curated free resources — books, videos, papers, courses, articles — with topic and type filtering. |
| **Glossary** | 83 flip cards. What people say vs what it actually means. |

### Claude Code integration

Each lesson has a `$ copy path` button that puts a ready-to-paste prompt on your clipboard:

```
I'm working on this lesson: phases/07-transformers-deep-dive/01-attention-mechanism
Read phases/07-transformers-deep-dive/01-attention-mechanism/docs/en.md and help me work through it.
```

Paste directly into Claude Code. No hunting for file paths.

### Progress and gamification

Every lesson tick earns XP. Level cost rises each level. Seven ranks unlock across 20 levels:

```
LV.01–03  Initiate          LV.10–12  AI Engineer
LV.04–06  Practitioner      LV.13–15  Senior Engineer
LV.07–09  Apprentice Eng.   LV.16–18  Staff Engineer
                            LV.19+    AI Architect
```

`game.js` is a pure function — curriculum + progress in, all stats out. No DOM, no side effects.

### WordPress integration

Progress persists to `/wp-json/aischool/v1/progress` when a nonce is present. Falls back to localStorage silently. Zero screen code changes when switching backends.

---

## Curriculum chapters

| Chapter | What you implement |
|---------|-------------------|
| 00 Setup & Tooling | Dev environment, GPU setup, APIs, Python toolchain |
| 01 Math Foundations | Linear algebra, calculus, probability, PCA, variance decomposition |
| 02 ML Fundamentals | Regression, classification, loss surfaces, gradient descent from scratch |
| 03 Deep Learning Core | Backprop by hand, activations, regularisation |
| 04 Computer Vision | CNNs, object detection, full vision pipeline |
| 05 NLP | Tokenisation, embeddings, sequence models |
| 06 Speech & Audio | Spectrograms, ASR, audio feature extraction |
| 07 Transformers Deep Dive | Attention, multi-head, positional encoding — transformer from scratch |
| 08 Generative AI | GANs, diffusion, CLIP |
| 09 Reinforcement Learning | Policy gradients, Q-learning, environment loops |
| 10 LLMs from Scratch | Pre-training, tokenisation, scaling laws |
| 11 LLM Engineering | Fine-tuning, RLHF, evaluation, inference optimisation |
| 12 Multimodal AI | Vision-language models, multimodal agent |
| 13 Tools & Protocols | MCP, tool use, function calling |
| 14 Agent Engineering | ReAct, planning, memory, agent workbench |
| 15 Autonomous Systems | Long-horizon agents, evaluation frameworks |
| 16 Multi-Agent & Swarms | Coordination, parallelism, swarm patterns |
| 17 Infrastructure & Production | Serving, monitoring, deployment pipelines |
| 18 Ethics, Safety & Alignment | Red-teaming, constitutional AI, safety evals |
| 19 Capstone Projects | 17 final builds that combine concepts across all phases |

**Languages:** Python · TypeScript · Rust · Julia

---

## Engineering decisions

**No framework.** Files open directly in a browser. No node_modules, no build step, no config. Personal tooling should outlive framework churn.

**Single source of truth.** `PHASES` is the curriculum. Every page derives its UI from `game.derive(PHASES, store.load())`. Adding a stat means editing data, not components.

**Adapter pattern.** localStorage and WordPress REST implement the same three-method interface. Swapping backends touches zero screen code.

**Pure game engine.** `game.js` works in a browser, a test runner, or Node — same function, same output, no imports.

---

## Run it

```bash
cd site-new
python3 -m http.server 8080
# open http://localhost:8080
```

No install. No build. No config.

---

## Credits

| | |
|--|--|
| [rohitg00/ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch) | Original curriculum — lesson content, phase architecture, six-beat lesson format |
| [EbookFoundation/free-programming-books](https://github.com/EbookFoundation/free-programming-books) | Library curation source |
| [Synaptic Labs / Professor Synapse](https://github.com/ProfSynapse) | Prompt methodology and AI engineering guidance |
| [garrytan/gstack / ICM](https://github.com/garrytan/gstack) | Toolchain and workspace methodology used throughout development |

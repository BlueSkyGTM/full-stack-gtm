<!-- Agent: Lyra-code -->
# Stage 06: Site Readability Overhaul

Implement modular lesson components without breaking the existing render pipeline.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Design system snapshot | `../00-a-curriculum-archaeology/output/design-system-snapshot.md` | Full file | Existing patterns to extend |
| Hybrid lessons | `../02-lesson-injection/output/hybrid-lessons/` | Sample lessons | Before/after reference |
| Render pipeline | `{{REPO_URL}}` | ui.js, catalog.js | Do not break these |
| Lyra code brief | `../00-c-agent-setup/output/agent-briefs/lyra-code-brief.md` | Full file | Site architecture constraints |

## Process

1. Run /design-review on current site — 80-item audit
2. Prioritize: lessons that look thin, unreadable, or not significant
3. Implement modular lesson components — do not break the existing render pipeline
4. Validate in browser before /review gate

## Audit

| Check | Pass Condition |
|-------|---------------|
| Render pipeline intact | ui.js and catalog.js behavior unchanged |
| Lessons look significant | No lesson renders as a thin block of text |
| Components are modular | Each component can be updated independently |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| `site-components/` | `output/` | Updated lesson HTML components |

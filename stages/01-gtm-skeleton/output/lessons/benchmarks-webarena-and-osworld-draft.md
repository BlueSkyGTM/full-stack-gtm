# Benchmarks: WebArena and OSWorld

## Beat 1: Hook
Agent vendors claim their models "can use a computer." Benchmarks are how you call bluff. WebArena and OSWorld are the two suites that turn "it works on my machine" into a reproducible number.

## Beat 2: Concept
Define what a web-agent benchmark measures: task completion against a held-out set of realistic workflows. Cover WebArena's architecture (standalone web instances for e-commerce, CMS, GitLab) and OSWorld's architecture (full Ubuntu desktop with real applications). Compare evaluation rubrics: exact-match vs. function-based vs. human-graded.

## Beat 3: Demo
Walk through a single WebArena task JSON: the initial state, the intent, the evaluation function. Then do the same for an OSWorld task. Show what "success" actually means in each framework by printing the evaluation criteria for one task from each suite.

## Beat 4: Use It
Connect agent benchmark scores to GTM automation decisions: if an agent scores below X% on WebArena form-filling tasks, it cannot be trusted to automate CRM data entry or LinkedIn navigation without human-in-the-loop. Foundational for Zone 2 (Enrichment) agent-based enrichment workflows. [CITATION NEEDED — concept: agent reliability thresholds for GTM task delegation]

## Beat 5: Ship It
- **Easy**: Fetch and print the task distribution across domains for WebArena's test set.
- **Medium**: Run a single WebArena task through a multimodal model and score the result against the evaluation function.
- **Hard**: Build a minimal evaluator that takes a trajectory (screenshot + action pairs) and replays it against an OSWorld-style checklist.

## Beat 6: Evaluate
Assessment targets: distinguish WebArena from OSWorld by task scope and environment, explain what a benchmark success rate of 15% on WebArena means for production viability, compare exact-match evaluation vs. function-based evaluation, and identify which GTM workflows are safe to automate at current SOTA agent scores.
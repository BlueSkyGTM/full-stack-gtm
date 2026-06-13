# Automated Alignment Research (Anthropic AAR)

## Hook
Anthropic's core bet: the most viable path to solving alignment is building AI systems that can *do* alignment research themselves. This beat introduces the alignment tax problem and why manual human oversight doesn't scale with capability.

## Concept
Mechanism breakdown of the AAR loop: train a model to generate alignment research proposals, evaluate them via automated checks + human raters, feed results back into training. Covers the three pillars Anthropic identifies—scalable oversight, mechanistic interpretability, and adversarial testing—as parallel tracks an AAR system would pursue. [CITATION NEEDED — concept: Anthropic AAR three-pillar framework exact taxonomy]

## Demo
Working Python script that simulates a simplified AAR evaluation pipeline: generates candidate "alignment strategies" from a prompt, scores them against a rubric using a second model call, ranks and filters, then prints the top candidates with scores. Observable output: ranked list of strategies with numerical scores.

## Use It
GTM redirect to **Zone 01 – Research & Ideation** and **ICP Scoring**. The AAR pattern—generate candidates, score against a rubric, filter—is structurally identical to automated ICP scoring and account research prioritization in Clay. The mechanism is: LLM generates → LLM evaluates → ranked output. Exercise hook: *Easy* — implement a two-call generate-then-score loop for company descriptions. *Medium* — build a rubric-based ICP scorer that outputs ranked accounts.

## Ship It
Build a working generate-evaluate-rank pipeline that takes a list of target accounts, generates fit narratives, scores them against a configurable rubric, and outputs a prioritized CSV. This is the AAR pattern applied to account research. Exercise hook: *Hard* — implement the full pipeline with configurable rubric, multiple scoring passes, and threshold-based filtering.

## Deep Dive
Examine what Anthropic actually published on AAR: the timeline assumptions (when AAR systems become viable), the bootstrapping problem (a misaligned AAR produces misaligned research), and the open question of whether automated interpretability can scale faster than capability. Includes Skeptical note: Anthropic has not published a working AAR system—this is a stated research direction, not a deployed artifact. [CITATION NEEDED — concept: Anthropic published timeline estimates for viable AAR]
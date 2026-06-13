# Reward Hacking and Goodhart's Law

## Hook

A 60-second setup: you launch an email scoring model that optimizes for reply rate. Within a week, the model discovers that sending one-word subject lines ("hey?") triples reply rate. Reply rate went up. Pipeline did not. This is Goodhart's Law in motion, and it is the central failure mode of any metric-driven system.

## Concept

Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure." Distinguish four categories of Goodhart effects — regressional, extremal, causal, and adversarial — and map each to real failure modes in automated GTM systems where proxy metrics (open rate, SQL count, ICP score) diverge from the outcome you actually want (revenue, retention).

## Mechanism

Formalize the gap: true objective `V` vs. proxy reward `R = V + ε`, where `ε` is noise or bias. When you optimize `R` instead of `V`, the agent exploits the residual. Walk through three escalation levels: (1) specification gaming — the agent satisfies the letter of the reward, (2) reward tampering — the agent modifies the reward source itself, (3) wireheading — the agent shortcuts the entire environment. For each, name the structural condition that enables it (observable proxy, modifiable reward function, or ungrounded feedback loop).

## Code

Build a minimal gridworld where an agent trained with reward `R` discovers a shortcut that maximizes `R` while tanking the true objective `V`. Print both `R` and `V` at each episode to show the divergence in real time. No RL framework dependency — pure Python loop with observable output.

Exercise hooks:
- **Easy**: Modify the reward function to close the original exploit; observe whether a new exploit emerges.
- **Medium**: Add a second proxy metric and compare — does averaging two proxies delay or prevent hacking?
- **Hard**: Implement a penalty term that detects distributional shift between training reward and deployment reward; print a warning when hacking is detected.

## Use It

GTM cluster: **Zone 1 — ICP & Scoring** and **Zone 2 — Signal Enrichment**. Every scoring model, lead grade, and intent signal is a proxy for "will this account close." Map the four Goodhart categories to concrete GTM failures: regressional (your ICP score has measurement error and you over-index on borderline accounts), extremal (you optimize past the training distribution into accounts the model has never seen), causal (you select accounts *because* they score high, but the score caused the selection, not the account quality), and adversarial (prospects or internal teams learn the scoring rubric and game it). Build a metric audit checklist: for each KPI your team targets, name the true outcome, name the proxy, and write down one way the proxy can be gamed.

Exercise hooks:
- **Easy**: Pick three GTM metrics your team uses. For each, write the true objective and one exploit path.
- **Medium**: Audit an existing Clay waterfall scoring model — identify which enrichment fields are proxy vs. ground-truth signals and flag the weakest proxy.
- **Hard**: Design a dual-metric system where two independent proxies must agree before action is taken. Implement it as a Clay formula.

## Ship It

Build a "Goodhart audit" artifact: a single-page document listing every automated metric in your GTM stack, its proxy–objective gap, and a scheduled review cadence. Ship it to your RevOps team with one concrete recommendation: replace one gamed metric with a harder-to-game alternative, or add a second uncorrelated proxy as a guard rail.

Exercise hooks:
- **Easy**: Write the audit for three metrics and share it with one teammate for feedback.
- **Medium**: Implement a Clay formula that flags accounts where two independent scoring signals disagree by more than a threshold.
- **Hard**: Set up a weekly automated report that compares proxy metric trend (e.g., SQL count) against true outcome trend (e.g., closed-won count) and surfaces divergence as an alert.
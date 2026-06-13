# Bayes' Theorem

## Beat 1: Hook
You already reason like a Bayesian — you just don't write it down. When a lead from a Fortune 500 company visits your pricing page twice in one week, you update your estimate of whether they'll buy. Bayes' Theorem is the mechanism that makes that update precise. Every lead score, every intent signal, every "hot or not" classification runs on this pattern whether the tool calls it Bayesian or not.

## Beat 2: Concept
Break down the theorem as an update rule: prior × likelihood ÷ evidence = posterior. Explain each term in plain language. Show how the posterior from one update becomes the prior for the next — this is the mechanism behind sequential signal processing in enrichment waterfalls and scoring models. Contrast with frequentist thinking (fixed rates vs. updating beliefs).

## Beat 3: Demonstration
Implement Bayes' Theorem from scratch in Python. Use a concrete worked example: given a base conversion rate (prior), a signal like "visited pricing page" with known lift (likelihood), compute the updated conversion probability (posterior). Print each term so the practitioner can trace the arithmetic. Then chain two signals together to show posterior-to-prior propagation.

## Beat 4: Use It
GTM redirect: this is the scoring mechanism behind Zone 1 (ICP Fit) and Zone 2 (Signal Detection). In practice, enrichment tools and scoring models apply this pattern — a Naive Bayes classifier scores leads by treating each intent signal as evidence that updates the probability the account matches ICP. [CITATION NEEDED — concept: specific enrichment tool using Naive Bayes for lead scoring]. Exercise hooks: Easy — compute posterior for a single signal given provided prior and likelihood. Medium — build a function that chains N signals sequentially. Hard — implement a Naive Bayes classifier on a small synthetic dataset of firmographic + intent features and evaluate precision/recall.

## Beat 5: Ship It
Build a command-line Bayesian lead scorer that takes a CSV of accounts with signals and outputs a ranked list with posterior conversion probabilities. The script loads priors from a config, ingests signal data, applies the update rule per signal per account, and writes scored output. This is the same pattern enrichment waterfalls use to stack signal confidence. Exercise hooks: Easy — score 10 accounts with 2 signals using hardcoded priors. Medium — load priors and signal lift rates from a JSON config file, score any number of accounts. Hard — add a calibration step that compares predicted posteriors against actual conversion data and adjusts priors automatically.

## Beat 6: Evaluate
Quiz questions testing: (1) identify which term is the prior vs. posterior in a scenario, (2) compute a posterior given numbers, (3) explain what happens to the posterior when the likelihood ratio approaches 1, (4) diagnose what goes wrong when signals are not conditionally independent in a Naive Bayes model. All questions drawn directly from the code and mechanisms demonstrated in Beats 3–5.
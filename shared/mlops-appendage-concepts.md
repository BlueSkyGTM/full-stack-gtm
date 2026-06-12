CURRICULUM CONCEPT NOTE
MLOps Appendage — What Comes After
What This Is
This is not a new track. It is a four-lesson appendage sitting at the end of the curriculum,
after Phase 17 (Infrastructure & Production). The AI engineering curriculum already
teaches you to build algorithms and ship them. This appendage covers what happens
once they are live — the operational layer that turns a one-time model into a living
system.
Source material: Made With ML by Goku Mohandas (madewithml.com), now backed by
Anyscale. Widely considered the gold standard for production ML education. 40K+
developers.
Why It Belongs Here
The curriculum already covers building and deploying. MLOps is what comes after:
what happens when your model is running in production and the world changes around
it. For a GTM engineer positioning themselves as an AI engineer, this is the layer that
most people skip — and therefore the layer that most differentiates.
The four concepts this appendage adds:
– Experiment tracking — knowing which model configuration worked and being able
to reproduce it
– Data versioning — treating training data like code so you can debug performance
changes
– Drift monitoring — detecting when your model is silently degrading before it costs
pipeline
– Retraining pipelines — automating the loop so the model gets better as new data
arrives
The Four Lessons
MLOps 01 — Experiment Tracking
Course URL: madewithml.com/courses/mlops/experiment-tracking

GitHub file: notebooks/madewithml.ipynb
What it teaches:
– How to log every experiment's parameters, metrics, and artifacts to a central
registry using MLflow
– How to compare runs side-by-side to identify which configuration produced the
best model
– How to load the best checkpoint from a tracked run for evaluation or deployment
– Tool options: MLflow (open source, self-hosted), W&B, Comet ML, Neptune
GTM application:
– When testing 3 lead scoring models or 6 prompt variants for outreach
personalization, MLflow gives you a systematic record of what changed and why
one worked.
Without this you just know "this one was better" — you cannot reproduce it or explain
it to a client.
– Tracking prompt experiments across Clay and GPT runs: which personalization
template produced the highest reply rate at which cost per credit
Outside tool bridge:
– W&B already integrates with Hugging Face fine-tuning runs. Students can use this
directly on Phase 11 LLM fine-tuning projects — real projects, real tracking, no
extra setup.
MLOps 02 — Data Versioning
Course URL: madewithml.com/courses/mlops/versioning
GitHub file: madewithml/data.py
What it teaches:
– How to version datasets the same way you version code using DVC so every model
run is tied to an exact snapshot of the data it trained on
– How to reproduce any past experiment by checking out its data version alongside
its code version
– How to isolate whether a model's performance changed because of data or code
GTM application:

– When a Clay enrichment waterfall changes — a new provider is added, a field
shifts, a source goes stale — DVC tells you whether your lead scoring model
degraded because of the data or something else.
This is the debugging layer. Without it you are guessing why your model stopped
working.
– Versioning the ICP training dataset: each time you add new closed-won data from
the CRM, you create a new version rather than overwriting the old one
– Enables rollback: if a new Clay provider introduces noise, you can revert to the last
clean data version and retrain
Outside tool bridge:
– Students can version their Clay-exported lead lists directly in DVC — real GTM
data, real versioning workflow, no artificial project needed.
MLOps 03 — Monitoring & Drift Detection
Course URL: madewithml.com/courses/mlops/monitoring
GitHub file: monitoring-ml/monitoring.ipynb
What it teaches:
– Three types of drift: data drift (input distribution shifts), target drift (labels shift),
concept drift (the relationship between inputs and outputs changes)
– How to measure drift statistically: univariate tests per feature, multivariate tests
across feature sets
– Alert, Inspect, Act: the three-step response protocol when drift is detected
– Tools: Evidently AI for drift reports, Great Expectations for data quality assertions
GTM application:
– A lead scoring model trained in January silently degrades by April when the ICP
shifts. Companies that were strong signals are no longer converting, but the model
still scores them high.
Drift detection catches this before it costs pipeline. Without it you find out from a
missed quarter.
– Data drift in GTM: a new job title appears in the market that the enrichment model
has never seen — the input distribution has shifted
– Concept drift in GTM: the relationship between "has a Salesforce CRM" and "is a
good fit" changes when your ICP evolves

Outside tool bridge:
– Evidently AI runs on any DataFrame. Students can feed it Clay-exported lead data
directly and generate drift reports without any ML infrastructure. Immediately
applicable, no course scaffolding required.
MLOps 04 — Retraining Pipelines & CI/CD for Models
Course URL: madewithml.com/courses/mlops/cicd +
madewithml.com/courses/mlops/jobs-and-services
GitHub file: .github/workflows/workloads.yaml
What it teaches:
– How to automate the full loop: new data arrives, model retrains, gets evaluated
against a performance threshold, gets promoted to production if it passes
– CI/CD for models using GitHub Actions: trigger training jobs on schedule or on
data change, run evaluation checks, deploy only if performance improves
– Jobs vs services: batch retraining (scheduled) vs always-on inference (real-time)
GTM application:
– This is what turns a one-time algorithm into a living system. The difference
between a lead scoring model you built once and one that gets better as your CRM
accumulates closed-won data.
This is the compounding advantage. Most GTM engineers stop at deployment.
Retraining pipelines are what separates systems thinkers from builders.
– Practical GTM loop: every Monday, HubSpot exports new closed-won and lost-deal
data, triggers a retraining job, evaluates the new model, promotes it automatically
if it outperforms — no manual work
– Performance gate: the model only gets promoted if its precision on your ICP
definition exceeds the previous version, preventing a bad data batch from silently
degrading the pipeline
Outside tool bridge:
– GitHub Actions is free. Students can build a weekly retraining workflow that pulls
from a HubSpot export CSV, retrains a sklearn scoring model, and logs the result
to MLflow — entirely with free tools, no infrastructure required. Real GTM
pipeline, real automation.

A Note on Flexibility
The Made With ML course uses Ray and Anyscale infrastructure for its examples. This is
heavier than what a GTM engineer needs. Professor Synapse should treat the conceptual
content as canonical and the specific tooling as flexible — students should be
encouraged to apply the patterns to lighter tools where possible (sklearn instead of
PyTorch, MLflow local instead of Anyscale, GitHub Actions instead of Kubeflow).
The goal is not to make GTM engineers into MLOps specialists. The goal is to make
them fluent enough that they can say: I don't just build the model, I monitor it, version
it, and retrain it automatically when performance degrades. That sentence is worth
significantly more in an interview than knowing how to configure Kubeflow.
Placement in Curriculum
– Sits after Phase 17 (Infrastructure & Production) in the AI engineering curriculum
– Sits after Phase 13 (Production GTM Infrastructure) in the GTM curriculum
– Labeled as an appendage, not a track — it does not interrupt the main curriculum
flow
– Professor Synapse should introduce it with explicit framing: "You have built the
model. You have shipped it. Now here is what keeps it alive."
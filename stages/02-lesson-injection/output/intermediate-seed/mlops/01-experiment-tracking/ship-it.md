## Ship It

Tracking records what happened. A registry declares what should happen next. The transition from experiment tracking to model registry is the transition from "I ran these experiments and here are the results" to "this specific model version is the one we serve to production." The registry is a versioned store with status labels — Staging, Production, Archived — that your serving infrastructure reads to decide which model to load.

The mechanism is straightforward. You take a run from the tracking server, register its model artifact under a named model entry, and the registry assigns it a version number starting at 1. Each version can be transitioned between stages. Multiple versions can exist simultaneously — you might have version 3 in Production and version 4 in Staging for evaluation. The registry does not care about training; it cares about deployment state.

Here is a script that registers the best run from the lead scoring experiment and transitions it through stages:

```python
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()
experiment = client.get_experiment_by_name("icp_lead_scoring")
runs = client.search_runs(
    [experiment.experiment_id],
    order_by=["metrics.f1_score DESC"],
    max_results=1,
)

best_run = runs[0]
model_uri = f"runs:/{best_run.info.run_id}/model"
model_name = "icp_lead_scoring_model"

mv = mlflow.register_model(model_uri, model_name)
print(f"Registered: {model_name} version {mv.version}")
print(f"Source run: {best_run.info.run_id[:8]}")
print(f"F1 score: {best_run.data.metrics.get('f1_score', '?')}")
print(f"Feature set: {best_run.data.params.get('feature_set', '?')}")

client.transition_model_version_stage(
    name=model_name,
    version=mv.version,
    stage="Staging",
)
print(f"Stage: Staging")

client.transition_model_version_stage(
    name=model_name,
    version=mv.version,
    stage="Production",
)
print(f"Stage: Production")

print("\nAll registered versions:")
versions = client.search_model_versions(f"name='{model_name}'")
for v in sorted(versions, key=lambda x: int(x.version)):
    print(
        f"  v{v.version} | stage={v.current_stage} | "
        f"run={v.run_id[:8]}"
    )
```

In a GTM context, the model you transition to Production is the one that scores every inbound lead. If your CRM or Clay workflow calls a model endpoint, that endpoint reads the registry's Production stage to determine which version to serve. When you train a better model next month, you register it as a new version, test it in Staging, and transition it to Production — the serving layer picks up the change without code deployment. [CITATION NEEDED — concept: CRM/Clay integration with model serving endpoints]
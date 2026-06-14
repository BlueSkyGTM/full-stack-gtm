## Ship It

Deploying a perceptron means two things: saving the learned weights and loading them at inference time to classify new inputs. The pattern is the same one you would use for any model: train once, serialize the parameters, load them in a separate script or service that handles prediction requests.

The perceptron has no runtime dependencies beyond NumPy. You serialize a dictionary of weights and bias to JSON. At inference time, you load the JSON, assert that the input vector has the same dimensionality as the training data, compute the dot product, and apply the step function. There is no model framework, no runtime, no serving infrastructure. This is as lightweight as inference gets.

```python
import json
import numpy as np

w = np.array([-0.842, 1.234, 0.456])
b = -0.078
feature_names = ["employee_count_norm", "funding_amount_norm", "uses_competitor_norm"]
means = np.array([174.0, 16.05, 0.4])
stds = np.array([164.06, 16.14, 0.49])

model = {
    "weights": w.tolist(),
    "bias": b,
    "feature_names": feature_names,
    "means": means.tolist(),
    "stds": stds.tolist(),
}

with open("perceptron_weights.json", "w") as f:
    json.dump(model, f, indent=2)
print("Saved weights to perceptron_weights.json")
```

Now the inference script — a separate file that knows nothing about training, only about loading weights and classifying:

```python
import json
import numpy as np

with open("perceptron_weights.json") as f:
    model = json.load(f)

w = np.array(model["weights"])
b = model["bias"]
means = np.array(model["means"])
stds = np.array(model["stds"])

def classify(raw_input):
    assert len(raw_input) == len(w), f"Expected {len(w)} features, got {len(raw_input)}"
    x_norm = (np.array(raw_input) - means) / stds
    z = np.dot(w, x_norm) + b
    return 1 if z >= 0 else 0, z

batch = [
    [250, 30.0, 1],
    [40, 1.0, 0],
    [500, 50.0, 1],
]

print(f"{'Lead':<20} {'Decision':<20} {'z-score':>10}")
print("-" * 52)
for lead in batch:
    pred, z = classify(lead)
    decision = "ROUTE" if pred == 1 else "DO NOT ROUTE"
    print(f"{str(lead):<20} {decision:<20} {z:>10.4f}")
```

The assert on input dimensionality is not decorative. In a GTM pipeline, feature drift is real: someone adds a new enrichment column, changes the order of fields in the CRM export, or a data provider changes their schema. The perceptron will silently produce garbage if the input dimensions silently shift. The assertion turns a silent failure into a loud one — which is the difference between a bug you catch in testing and a routing decision that sends every lead to the wrong destination.

The harder limitation is the absence of confidence calibration. The perceptron outputs 0 or 1 — nothing in between. A lead with `z = 0.01` gets the same "ROUTE" label as a lead with `z = 15.0`. In a GTM context, this means you cannot tier leads by warmth, cannot set a "review queue" threshold below the auto-route threshold, and cannot prioritize within the routed bucket. Every routed lead looks identical. This is acceptable for a simple binary filter — "is this lead even worth looking at?" — but insufficient for any workflow that needs to rank or prioritize. When you need probabilities, you need a sigmoid activation, which is logistic regression. That is the next lesson.
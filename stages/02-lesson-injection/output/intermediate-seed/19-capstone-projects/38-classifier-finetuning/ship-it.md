## Ship It

Deploying the head-swap model means saving it to disk and loading it in an inference script or service. The `transformers` library handles serialization through `save_pretrained()`, which writes both the backbone and the trained head to a single directory.

```python
import os

SAVE_DIR = "./intent_classifier_frozen"
model_frozen.save_pretrained
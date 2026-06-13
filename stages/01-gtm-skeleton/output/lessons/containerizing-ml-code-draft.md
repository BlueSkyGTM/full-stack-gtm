# Containerizing ML Code

---

## Hook It

The "works on my machine" problem multiplies in ML: CUDA version, Python version, OS-level libraries, and model weights must all align. A container freezes the entire dependency graph into a portable unit that runs identically on a laptop, a CI runner, or a GPU node.

---

## Ground It

**Prerequisite check:** The student should have written a Python script that loads a model and produces an inference, and should be comfortable in a terminal.

**Concepts in play:**
- **Container vs. VM:** A container shares the host kernel; a VM ships its own. Containers are lighter but offer less isolation.
- **Image vs. Container:** An image is the blueprint (filesystem + metadata). A container is a running instance of that image.
- **Layer caching:** Each `RUN`, `COPY`, or `ADD` creates a layer. Docker caches layers; unchanged layers are reused on rebuild. Order matters for build speed.
- **Namespaces and cgroups** (Linux primitives): Namespaces isolate what the process can see (filesystem, network). Cgroups limit what it can use (CPU, memory). Docker wraps both.

**Why ML needs this more than web dev:** A Flask app might have 12 Python dependencies. An ML inference pipeline often has Python dependencies *plus* CUDA runtime *plus* system libraries like `libgomp` *plus* compiled extensions. The combinatorics explode.

---

## Build It

### Beat 1: Write a minimal Dockerfile for inference

**Mechanism:** Start from a base image that already includes the Python version and (if needed) CUDA runtime. Install Python dependencies. Copy model code. Set the entrypoint.

```python
# save as main.py
import json
import sys

def predict(features):
    return {"score": sum(features) / len(features)}

if __name__ == "__main__":
    features = json.loads(sys.argv[1])
    result = predict(features)
    print(json.dumps(result))
```

```dockerfile
# save as Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY main.py .

ENTRYPOINT ["python", "main.py"]
```

**Build and run:**
```bash
docker build -t ml-inference:v1 .
docker run ml-inference:v1 '[3.0, 5.0, 7.0]'
```

**Observable output:** `{"score": 5.0}`

### Beat 2: Layer ordering for fast rebuilds

**Mechanism:** `COPY` changes every time source code changes. `pip install` changes rarely. Put `pip install` before `COPY` of application code so dependency installation is cached across code edits.

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENTRYPOINT ["python", "main.py"]
```

**Exercise hook (easy):** Rebuild twice after changing only `main.py`. Observe that the `pip install` layer is cached (watch build time drop).

### Beat 3: Multi-stage builds for smaller images

**Mechanism:** Build dependencies in one stage, copy only the runtime artifacts to a slim final image. Reduces image size and attack surface.

```dockerfile
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY main.py .

ENTRYPOINT ["python", "main.py"]
```

**Exercise hook (medium):** Compare image sizes of single-stage vs. multi-stage builds using `docker images`. Report the difference in MB.

---

## Use It

**GTM redirect:** Containerized scoring models are deployable as HTTP microservices behind any orchestration layer. In the context of [CITATION NEEDED — concept: GTM Zone mapping for model serving infrastructure], a containerized inference endpoint receives a webhook from Clay, scores the payload, and returns an enrichment result. The container guarantees that the CUDA/Python/weights stack that worked in development is identical in production.

**Specific pattern:**
1. Wrap inference code in a FastAPI (or Flask) server inside the container.
2. Expose a `/predict` route.
3. Clay's HTTP integration (or any webhook) POSTs to that endpoint.
4. The container responds with a JSON payload that Clay ingests into the enrichment waterfall.

```python
# save as server.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class Payload(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(payload: Payload):
    score = sum(payload.features) / len(payload.features)
    return {"score": score}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .

EXPOSE 8000
CMD ["python", "server.py"]
```

```bash
docker build -t ml-server:v1 .
docker run -p 8000:8000 ml-server:v1
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [3.0, 5.0, 7.0]}'
```

**Observable output:** `{"score":5.0}`

**Exercise hook (hard):** Add a `/health` endpoint that returns model metadata (version, input schema). Wire it into a Docker `HEALTHCHECK` directive. Verify with `docker inspect` that the health status updates correctly.

---

## Ship It

**Production concerns:**

1. **Image tagging strategy:** Never deploy `:latest` in production. Use semantic versioning (`v1.2.3`) or git SHA tags. Tags are mutable; SHA digests (`sha256:...`) are immutable. Pin to digests for deterministic deploys.

2. **Registry:** Push to a private registry (ECR, GCR, Docker Hub private repo). Never ship proprietary model weights in a public image.

```bash
docker tag ml-server:v1 your-registry.example.com/ml-server:v1
docker push your-registry.example.com/ml-server:v1
```

3. **Resource limits:** ML containers can consume unbounded memory. Set limits at orchestration time:

```bash
docker run -m 2g --cpus=2 -p 8000:8000 ml-server:v1
```

4. **Security:** Run as a non-root user. Use `.dockerignore` to exclude `.git`, model checkpoints, and local env files.

```dockerfile
RUN useradd -m appuser
USER appuser
```

5. **Logging:** Write to stdout/stderr. Docker captures both. Do not write logs to files inside the container.

**Exercise hook (medium):** Build a production-grade image that: runs as non-root, has a `.dockerignore`, pins all dependency versions, and tags with the current git SHA. Verify the user is non-root by running `whoami` inside the container.

---

## Prove It

**Exercise hooks only:**

| Difficulty | Exercise |
|---|---|
| Easy | Rebuild the minimal Dockerfile three times with minor code changes. Report which layers cache and which rebuild. Explain why. |
| Medium | Containerize an existing ML script that uses `scikit-learn`. Handle the case where the model file (`.pkl`) is 500MB and must be downloaded at build time from a remote URL. Optimize layer caching. |
| Hard | Build a multi-stage Dockerfile that trains a small model in stage 1, copies the serialized model to stage 2, and serves it via FastAPI in stage 2. The final image must not contain training dependencies (numpy is fine; scikit-learn training utilities must be absent). Verify by inspecting installed packages in the final image. |

**Assessment objectives (testable):**
1. Write a Dockerfile that layers dependency installation before code copy for cache efficiency.
2. Detect and fix a Dockerfile where layer ordering causes unnecessary rebuilds.
3. Configure a container to run as a non-root user with memory limits.
4. Explain why multi-stage builds reduce final image size and what artifacts carry between stages.
5. Implement a containerized inference endpoint that accepts JSON and returns predictions over HTTP.
# Containerizing ML Code

> The model works perfectly on your laptop. It fails in CI with a CUDA version mismatch. It fails in staging because the GLIBC is too old. It fails in production because the colleague who deployed it installed numpy 1.24 and you trained with numpy 2.0. Docker does not make these problems impossible — it makes them visible and fixable once rather than debuggable forever.

**Type:** Build
**Languages:** Python, Dockerfile
**Prerequisites:** Phase 17/29 (MLflow), basic terminal familiarity
**Time:** ~75 minutes

## Learning Objectives

- Explain the difference between a Docker image and a container, and what the layer cache buys you in ML builds.
- Write a multi-stage Dockerfile that separates the build environment from the runtime environment.
- Use `.dockerignore` to exclude model weights, data, and virtual environments from the build context.
- Build and run a containerized FastAPI inference server, testing it with `curl`.

## The Problem

A trained model is not just a `.pkl` file. It is a specific version of Python, a specific version of PyTorch, a specific CUDA toolkit, a specific set of library versions that were compatible on the training machine. When you hand that file to anyone else — a colleague, a CI runner, a cloud VM — they have a different environment. "Works on my machine" is the most expensive four words in ML engineering.

Containers solve this by packaging the code, the runtime, and the dependencies into a single artifact. The same image that runs on a developer's laptop runs in CI, staging, and production. The environment is no longer a conversation — it is a file.

## The Concept

### Images and containers

A **Docker image** is an immutable filesystem snapshot. It contains an operating system base, installed packages, and your application code. An image is built once from a `Dockerfile` and stored in a registry (Docker Hub, ECR, GHCR).

A **container** is a running instance of an image. You can run multiple containers from the same image simultaneously. Containers are isolated from each other and from the host OS at the process and filesystem level.

The relationship: `Dockerfile` → `docker build` → image → `docker run` → container.

### The layer cache

Docker builds images as a stack of layers. Each instruction in a `Dockerfile` that modifies the filesystem produces a new layer. Layers are cached by the build system and reused on subsequent builds if the instruction and all preceding instructions are unchanged.

In ML builds, layer ordering matters enormously:

```dockerfile
# Bad: requirements.txt changes every time code changes
COPY . /app
RUN pip install -r /app/requirements.txt   # cache misses every code change

# Good: dependencies change rarely; code changes often
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt   # cached unless requirements change
COPY . /app                                 # only this layer misses on code change
```

A cold Python ML image build can take 5-10 minutes. With correct layer ordering, incremental rebuilds take under 30 seconds.

### Multi-stage builds

A multi-stage Dockerfile uses multiple `FROM` instructions. Earlier stages build and compile; later stages copy only the artifacts they need. The final image contains only the runtime, not the build tools.

```dockerfile
# Stage 1: build environment with compilers
FROM python:3.12-slim AS builder
RUN pip install --prefix=/install -r requirements.txt

# Stage 2: runtime — no pip, no compilers, no build cache
FROM python:3.12-slim AS runtime
COPY --from=builder /install /usr/local
COPY app/ /app/
CMD ["python", "/app/serve.py"]
```

The final runtime image is typically 60-80% smaller than a naive single-stage image. Smaller images pull faster, scan faster, and have a smaller attack surface.

### .dockerignore

The build context is everything Docker sends to the daemon before building. Without `.dockerignore`, `docker build .` ships your entire project directory — including 10GB of model weights, your Python virtual environment, and your git history — to the daemon before the first instruction runs.

```
# .dockerignore
.git/
.venv/
__pycache__/
*.pyc
data/
models/            # weights go in volumes or object storage, not images
mlruns/
*.egg-info/
```

A correct `.dockerignore` keeps the build context under 1MB. Without it, ML builds routinely exceed 10GB context transfer time.

### ML image base choices

| Base | Size | When |
|------|------|------|
| `python:3.12-slim` | ~150MB | CPU inference, no GPU |
| `pytorch/pytorch:2.x-cuda12.x-cudnn9-runtime` | ~6GB | GPU inference |
| `nvidia/cuda:12.x-runtime-ubuntu22.04` | ~1.5GB | Custom CUDA, minimal |

For CPU inference (API serving a cached model), use `python:slim`. It is smaller, faster to pull, and easier to audit. Only use CUDA base images when the container actually needs the GPU at runtime.

### Running and testing

```bash
# Build the image
docker build -t ml-server:latest .

# Run the container, mapping host port 8000 to container port 8000
docker run --rm -p 8000:8000 ml-server:latest

# Test the health endpoint from the host
curl http://localhost:8000/health

# Test the predict endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

### Environment variables and secrets

Never bake secrets into images. Pass them as environment variables at runtime:

```bash
docker run -e MLFLOW_TRACKING_URI=http://tracking:5000 ml-server:latest
```

Or use Docker secrets / Kubernetes secrets for production. The image itself must be publishable to a public registry without exposing credentials.

## Build It

Build a multi-stage Dockerfile for a minimal FastAPI inference server that loads a pre-trained Scikit-learn model and serves predictions at `/predict`.

The code in `code/` contains:
- `app.py` — FastAPI server with `/health` and `/predict` endpoints
- `train_and_save.py` — trains the Iris classifier and saves it as `model.pkl`
- `Dockerfile` — multi-stage build for the inference server
- `.dockerignore` — excludes data, weights, and build artifacts

Run sequence:
```bash
cd code/
python train_and_save.py          # writes model.pkl
docker build -t iris-server .     # multi-stage build
docker run --rm -p 8000:8000 iris-server
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

The tests validate the FastAPI app directly (no Docker daemon required) by importing the app and using `TestClient`.

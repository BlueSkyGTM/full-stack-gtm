# Docker for AI

## Hook

You've got a model that works on your machine. The inference server dies on staging because the CUDA version mismatches by one minor revision. Docker solves this by packaging the entire runtime — libraries, CUDA toolkit, model weights, and serving code — into a reproducible unit. Without containers, AI deployment is environment gambling.

## Concept

**Layer 1: Container fundamentals for AI workloads.** A container is a namespace-isolated process that shares the host kernel but carries its own filesystem. For AI, this means you bundle Python, PyTorch/TensorFlow, CUDA libraries, and your model code into one immutable artifact. The mechanism: union filesystem layers stacked read-only with a writable layer on top. Each `RUN` in a Dockerfile creates a new layer.

**Layer 2: GPU passthrough.** Containers don't include GPUs — they access the host's GPU via the NVIDIA Container Toolkit (`nvidia-container-toolkit`). The mechanism: at container start, the runtime mounts the host's CUDA device files (`/dev/nvidia*`) and injects the matching CUDA libraries into the container's namespace. The container uses the host's GPU driver but its own CUDA toolkit version.

**Layer 3: Image sizing.** AI base images (`nvidia/cuda`, `pytorch/pytorch`) are 4–10 GB. Multi-stage builds compile/install dependencies in a "builder" stage, then copy only the runtime artifacts to a slim final image. The mechanism: each `FROM` starts a new stage; `COPY --from=<stage>` pulls artifacts between stages.

**Layer 4: Compose for AI stacks.** A typical AI serving stack is model server + vector DB + API gateway. Docker Compose declares each service, its GPU reservations, volume mounts for model weights, and network links. The mechanism: `docker compose` reads `compose.yaml`, creates a user-defined bridge network, and starts containers with inter-service DNS resolution by service name.

## Demonstration

Build and run a minimal AI inference container. The working code demonstrates:

1. **Dockerfile** that uses a CUDA base image, installs dependencies, and copies inference code
2. **GPU verification** — `nvidia-smi` inside the container confirming GPU access
3. **Volume mount** — model weights loaded from a host-mounted directory
4. **Docker Compose** — a two-service stack (FastAPI inference server + Redis for request queuing)

Every example produces observable output: print statements confirming model loaded, GPU device name, and inference result.

```python
# verify_gpu.py — run inside container to confirm GPU passthrough
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print(f"Tensor on device: {x.device}")
    print(f"Tensor values: {x}")
else:
    print("No GPU detected — running CPU-only")
```

Exercise hooks:
- **Easy:** Modify the Dockerfile to use a different CUDA base image and confirm GPU detection still works
- **Medium:** Add a multi-stage build that installs dependencies in a builder stage, then copies only the runtime to a slim image — compare final image sizes with `docker images`
- **Hard:** Write a `compose.yaml` that runs an inference server with GPU reservation and a health check endpoint, then verify the health check passes with `curl`

## Use It

In GTM Zone 3 (Enrichment), custom scoring models often need to run as internal APIs that Clay webhooks call. Docker containers let you deploy a FastAPI inference endpoint to any cloud VM or Kubernetes cluster with identical behavior. The specific pattern: containerize your lead-scoring model, expose it as an HTTP endpoint, and call it from Clay's HTTP integration or from n8n as an enrichment step. [CITATION NEEDED — concept: Clay webhook-to-custom-model enrichment pattern]

For local LLM usage in GTM workflows: containerize an Ollama or vLLM server with your chosen model, keep it running on an internal GPU instance, and use it for personalized email generation without sending data to external APIs.

Exercise hooks:
- **Easy:** Build a Docker image that serves a mock scoring API (`/score` returns `{"score": 0.87}`) and test it with `curl`
- **Medium:** Write a Dockerfile for a real sentiment analysis model (e.g., DistilBERT from HuggingFace), serve it via FastAPI, and hit it with a test request
- **Hard:** Deploy a two-service Compose stack (scoring model + Redis queue), send 100 requests, and confirm all responses return valid scores

## Ship It

Production deployment patterns for containerized AI:

1. **Image registry:** Push your built image to a registry (ECR, GCR, Docker Hub). CI/CD builds on push to main, tags with commit SHA.
2. **GPU instance deployment:** On cloud GPU VMs (AWS G5/G4, GCP with L4/A100), install Docker + NVIDIA Container Toolkit, pull the image, run with `--gpus all`.
3. **Health checks and restart policy:** `HEALTHCHECK` in Dockerfile + `--restart=unless-stopped` ensures the inference server recovers from crashes.
4. **Model weight management:** Store weights in a cloud bucket (S3/GCS), download at container start via entrypoint script, or pre-bake into the image for faster cold starts (trade-off: image size vs. startup time).

The anti-pattern to avoid: baking large model weights (2+ GB) into the image on every build. Instead, mount weights via volume or download at startup from a versioned bucket path.

Exercise hooks:
- **Easy:** Tag and push a Docker image to Docker Hub, then pull and run it on a different machine
- **Medium:** Write an entrypoint script that downloads model weights from S3 on first run and caches them in a mounted volume
- **Hard:** Set up a GitHub Actions workflow that builds your inference Docker image on push, runs integration tests against it using a Compose test stack, and pushes to ECR on passing tests

## Quiz

Questions grounded in the mechanisms above. Each question tests a specific learning objective:

1. **GPU passthrough mechanism:** "A container runs PyTorch inference. `torch.cuda.is_available()` returns False. The host has an NVIDIA GPU with drivers installed. What is the most likely missing component?" — Tests understanding of NVIDIA Container Toolkit requirement.
2. **Multi-stage build purpose:** "You have a Dockerfile where the builder stage is 12 GB (includes compilers, dev headers) and the final stage is 3 GB (runtime only). What mechanism produces this size reduction?" — Tests understanding of `COPY --from` and layer isolation.
3. **Volume mounts for model weights:** "Your team rebuilds the Docker image daily. Model weights are 4 GB and rarely change. Baking weights into the image causes 20-minute builds. What change reduces build time?" — Tests understanding of volume mounts vs. baked-in artifacts.
4. **Compose networking:** "In a `compose.yaml` with services `inference` (port 8000) and `redis` (port 6379), what URL does the inference service use to connect to Redis?" — Tests understanding of Compose DNS resolution by service name.
5. **CUDA version compatibility:** "Your Dockerfile uses `nvidia/cuda:11.8.0-runtime-ubuntu22.04` but the host has CUDA driver 525. The container fails with 'CUDA driver version is insufficient.' Is this a Docker problem, a driver problem, or a toolkit version problem?" — Tests understanding of driver vs. toolkit version matrix.
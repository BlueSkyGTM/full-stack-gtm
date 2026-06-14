# Docker for AI

## Learning Objectives

- Build a GPU-enabled Docker image with CUDA, PyTorch, and serving code from a Dockerfile
- Configure NVIDIA Container Toolkit to pass host GPUs into containers and verify access at runtime
- Mount host directories as volumes to persist model weights and datasets across container rebuilds
- Orchestrate multi-service AI stacks with Docker Compose using inter-service DNS resolution
- Compare image sizes between single-stage and multi-stage Dockerfiles and identify which layers dominate

## The Problem

You trained a model on your laptop with PyTorch 2.3, CUDA 12.4, and Python 3.12. The inference server dies on staging because the staging box has CUDA 12.1 and PyTorch 2.2. The error message is a stack trace referencing `libcudart.so.12` version mismatch — not something the model code can catch or fix. Your colleague's machine has a different GPU driver entirely. None of these failures are in your Python code. They are all in the gap between "my environment" and "the deployment target."

AI projects are dependency nightmares at three layers: the Python package layer (torch, transformers, flash-attn), the CUDA toolkit layer (cuDNN, cuBLAS, NCCL), and the system layer (glibc, NVIDIA driver, kernel modules). A version skew at any layer can crash inference silently or produce different outputs than training. The same model weights produce different logits on CUDA 11.8 versus 12.4 due to floating-point differences in the kernel implementations. This is not a theoretical concern — it is the reason your A/B test results don't match your offline evaluation.

Docker solves this by packaging the entire runtime stack — Python interpreter, CUDA libraries, system-level C dependencies, and your model code — into a single immutable artifact called an image. The image runs identically on every host that has the same kernel and NVIDIA driver. Everything else travels with the container.

## The Concept

### Layer 1: Container fundamentals for AI workloads

A container is a namespace-isolated process that shares the host kernel but carries its own filesystem, network stack, and process space. Unlike a virtual machine, there is no guest OS — the container uses the host's kernel directly, which is why it starts in under a second instead of minutes. For AI workloads, this means you bundle Python, PyTorch, CUDA runtime libraries, cuDNN, and your model code into one filesystem image.

The mechanism is a **union filesystem**. A Docker image is a stack of read-only layers. Each instruction in a Dockerfile (`FROM`, `RUN`, `COPY`) creates a new layer. When you run a container, Docker adds a thin writable layer on top. If your process writes to `/tmp/output.json`, that write goes into the writable layer and is lost when the container stops. This is why model weights need volume mounts — they live on the host filesystem, not in the container's ephemeral writable layer.

```mermaid
graph TD
    subgraph host["Host machine"]
        kernel["Linux kernel + NVIDIA driver"]
        subgraph container["Container (isolated namespaces)"]
            writable["Writable layer (ephemeral)"]
            ro3["Layer: COPY model code"]
            ro2["Layer: RUN pip install torch"]
            ro1["Layer: FROM nvidia/cuda:12.4"]
            writable --> ro3 --> ro2 --> ro1
        end
        volume["Host volume<br/>/home/user/models"]
        ro3 -.->|mounted at /models| volume
    end
    container -->|shares| kernel
```

### Layer 2: GPU passthrough

Containers do not contain GPUs. A container is just a Linux process with extra isolation flags — it has no hardware of its own. When a containerized PyTorch process calls `torch.cuda.is_available()`, it is checking whether the host's GPU is accessible from inside the container's namespace.

The NVIDIA Container Toolkit (`nvidia-container-toolkit`) is the bridge. At container start, the Docker runtime calls a hook provided by the toolkit. The hook mounts the host's NVIDIA device files (`/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`) into the container's `/dev` namespace. It also injects the matching driver libraries into the container's library path. The critical detail: the container uses the **host's GPU driver** (kernel module) but its **own CUDA toolkit version** (userspace libraries bundled in the image). This is why a CUDA 12.4 container runs on a host with driver version 550.x — the driver is shared, the toolkit is not.

You request GPU access in two ways. With plain Docker, you pass `--gpus all` or `--gpus '"device=0,1"'`. With Docker Compose, you declare a device reservation under `deploy.resources.reservations.devices`. Both tell the runtime to invoke the NVIDIA hook during container creation.

### Layer 3: Image sizing

AI base images are large. `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` is approximately 3.5 GB. Add PyTorch with CUDA support and you are at 6–8 GB. Add transformers, flash-attn (which compiles CUDA kernels at install time and needs `gcc`, `ninja-build`, and CUDA development headers), and you are at 10+ GB. Every `docker push` and `docker pull` moves that much data.

Multi-stage builds address this by separating the build environment from the runtime environment. You declare a `builder` stage with `FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` (the `-devel` tag includes compilers and headers), compile or install heavy dependencies, then copy only the artifacts to a final stage based on the `-runtime` tag (which omits compilers). The mechanism: each `FROM` starts a new build stage with its own layer stack. `COPY --from=builder /path /path` pulls files between stages without carrying the builder's layers into the final image.

### Layer 4: Compose for AI stacks

A real AI serving stack is rarely a single container. You have a model server (FastAPI, vLLM, TGI), a vector database (Qdrant, Milvus, pgvector), a cache or queue (Redis, RabbitMQ), and sometimes a GPU-side embedding service. Docker Compose declares each of these as a service in a single `compose.yaml` file.

The mechanism: `docker compose up` reads the file, creates a user-defined bridge network, starts each container on that network, and registers each service name as a DNS hostname. Your FastAPI container can reach Redis at `redis:6379` — no IP addresses, no environment variables for hostnames. You declare GPU reservations, volume mounts for model weights, port mappings, and inter-service dependencies (`depends_on`) in the same file.

## Build It

You are going to build a GPU-enabled inference container from scratch. Every piece here runs and produces observable output. You need Docker and the NVIDIA Container Toolkit installed on a machine with an NVIDIA GPU. If you do not have a GPU, the code still runs — it falls back to CPU and prints that fact.

Start with the project structure:

```
docker-ai/
├── Dockerfile
├── app.py
├── verify_gpu.py
├── compose.yaml
└── models/
    └── .gitkeep
```

The Dockerfile uses a CUDA base image, installs Python and PyTorch, and copies the inference code:

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update \
    && apt-get install -y python3.10 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN pip install --no-cache-dir \
    torch==2.3.0 \
    fastapi==0.111.0 \
    uvicorn[standard]==0.30.1 \
    pydantic==2.7.1

WORKDIR /app
COPY app.py verify_gpu.py ./

RUN mkdir -p /models

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

The GPU verification script. Run this inside the container to confirm that GPU passthrough is working and that PyTorch can see the device:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")

    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    y = x * 2
    print(f"Input tensor:  {x.cpu().tolist()}")
    print(f"Output tensor: {y.cpu().tolist()}")
    print(f"Tensor device: {x.device}")
else:
    print("No GPU detected — running CPU-only")
    x = torch.tensor([1.0, 2.0, 3.0])
    y = x * 2
    print(f"Input tensor:  {x.tolist()}")
    print(f"Output tensor: {y.tolist()}")
```

The FastAPI inference server. It loads a simple linear model on GPU at startup and serves predictions. The `/health` endpoint confirms GPU status so you can verify the container is wired correctly:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI(title="Docker AI Inference Server")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.nn.Sequential(
    torch.nn.Linear(10, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1),
).to(device)

model.eval()

torch.manual_seed(42)
with torch.no_grad():
    for param in model.parameters():
        param.add_(torch.randn_like(param) * 0.01)

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: float
    device: str

@app.get("/health")
def health():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return {
        "status": "ok",
        "device": str(device),
        "gpu": gpu_name,
        "cuda_available": torch.cuda.is_available(),
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    x = torch.tensor(request.features, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(x).squeeze()
    return PredictionResponse(prediction=float(output), device=str(device))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Build and run the GPU verification script:

```bash
docker build --target=verify -t ai-verify . -f- <<'EOF'
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS verify
RUN apt-get update && apt-get install -y python3.10 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.3.0
WORKDIR /app
COPY verify_gpu.py .
CMD ["python3", "verify_gpu.py"]
EOF

docker run --rm --gpus all ai-verify
```

Expected output:

```
PyTorch version: 2.3.0
CUDA available: True
GPU device: NVIDIA GeForce RTX 4090
CUDA version: 12.4
Device count: 1
Input tensor:  [1.0, 2.0, 3.0]
Output tensor: [2.0, 4.0, 6.0]
Tensor device: cuda:0
```

Now build the full inference server and test it:

```bash
docker build -t ai-inference .

docker run --rm --gpus all -p 8000:8000 -v $(pwd)/models:/models ai-inference
```

In another terminal, verify the server:

```bash
curl http://localhost:8000/health
```

Expected output:

```json
{"status":"ok","device":"cuda","gpu":"NVIDIA GeForce RTX 4090","cuda_available":true}
```

Test a prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}'
```

Expected output:

```json
{"prediction":0.03421689879894257,"device":"cuda"}
```

Finally, the Compose file that declares the inference server and Redis as a two-service stack:

```yaml
services:
  inference:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./models:/models
    depends_on:
      - redis
    environment:
      - REDIS_URL=redis://redis:6379

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

Start the full stack:

```bash
docker compose up --build
```

The inference service reaches Redis at `redis:6379` — that hostname resolves because Compose created a bridge network and registered both services on it. Verify both services are running:

```bash
docker compose ps
```

Expected output:

```
NAME                       IMAGE               STATUS         PORTS
docker-ai-inference-1      docker-ai-inference  Up             0.0.0.0:8000->8000/tcp
docker-ai-redis-1          redis:7-alpine      Up             0.0.0.0:6379->6379/tcp
```

## Use It

Containerized PyTorch neural network inference is the deployment pattern for lead-likelihood scoring in outbound infrastructure (GTM Infrastructure cluster, §1.4). The same Dockerfile and Compose pattern from Build It packages the model that scores whether a prospect fits your ICP. Enrichment data from Apollo, ZoomInfo, or 6sense flows in as feature vectors; the containerized model returns a score; a sync worker writes the result to Salesforce or HubSpot. The container is the boundary that prevents `requests==2.31` in the enrichment service from clashing with `requests==2.28` in the Salesforce SDK.

Create this `compose.gtm.yaml` alongside the Build It artifacts and run it:

```yaml
services:
  lead-scorer:
    build: .
    ports:
      - "8001:8000"
    volumes:
      - ./models:/models:ro
    environment:
      - MODEL_PATH=/models/lead_score_v2.pt
      - SCORE_THRESHOLD=0.72
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 5s
      timeout: 3s
      retries: 5
    restart: unless-stopped

  enrichment-webhook:
    image: python:3.11-slim
    command: >
      python -c "
      import urllib.request, json;
      data = json.dumps({'features': [0.8, 0.3, 0.1, 0.5, 0.9, 0.2, 0.7, 0.4, 0.6, 0.15]}).encode();
      req = urllib.request.Request('http://lead-scorer:8000/predict', data=data, headers={'Content-Type': 'application/json'});
      resp = urllib.request.urlopen(req);
      print('Lead score:', json.loads(resp.read()))
      "
    depends_on:
      lead-scorer:
        condition: service_healthy
```

```bash
docker compose -f compose.gtm.yaml up --build
```

The enrichment-webhook service resolves `lead-scorer` via Compose DNS — no hardcoded IP addresses. The `:ro` volume mount means you can swap `models/lead_score_v2.pt` on the host and restart the container to deploy a new model version without rebuilding the image. The healthcheck gates the webhook from firing predictions until the model server is ready. [CITATION NEEDED — concept: GTM Infrastructure cluster §1.4 containerized lead-scoring deployment patterns]

## Exercises

**Exercise 1: Swap the CUDA base.** Change the Dockerfile's `FROM` line from `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` to `nvidia/cuda:12.1.1-cudnn-runtime-ubuntu22.04`. Rebuild, run `verify_gpu.py`, and observe what changes. Does PyTorch still install without errors? Does `torch.cuda.is_available()` return True? What CUDA version does `torch.version.cuda` report? Run `docker images` before and after to capture the image size difference — write it down.

**Exercise 2: Add a volume for model weights.** Create a file `models/config.json` on your host with `{"model_name": "linear-test", "version": "0.1"}`. Modify `app.py` to read this file at startup and include the model name and version in the `/health` response. Rebuild and run with `-v $(pwd)/models:/models`. Confirm the health endpoint returns the config. Then change the JSON on the host without rebuilding the image, restart the container with `docker restart`, and confirm the new config is picked up — this is the hot-swap pattern you will use to deploy new model versions in production.

## Key Terms

**Container** — A namespace-isolated Linux process that shares the host kernel but carries its own filesystem, network, and process space. Starts in under a second because there is no guest OS boot.

**Image** — A read-only stack of filesystem layers. The build artifact. Each Dockerfile instruction creates one layer. Images are immutable — you tag new versions, you do not edit existing ones.

**Layer** — A filesystem diff produced by one Dockerfile instruction. Stored as a tar archive. Layers are shared across images and deduplicated by the registry during push and pull.

**Union filesystem** — The mechanism that stacks read-only layers and presents them as a single filesystem to the container. OverlayFS is the default on modern Docker installations.

**NVIDIA Container Toolkit** — A set of hooks that the Docker runtime calls at container start to mount host GPU device files and driver libraries into the container's namespace. The container uses the host's kernel driver but its own bundled CUDA userspace libraries.

**Volume mount** — A host directory or Docker-managed volume mapped into the container's filesystem. Bypasses the union filesystem — reads and writes go directly to the host. Used for model weights, datasets, and output persistence across container restarts.

**Multi-stage build** — A Dockerfile pattern that uses multiple `FROM` instructions to separate the build environment (compilers, dev headers) from the runtime image. `COPY --from=builder` pulls artifacts between stages without carrying the builder's layers into the final image.

**Docker Compose** — A declarative YAML format for defining multi-service stacks. Creates a user-defined bridge network and registers each service name as a DNS hostname, so services reach each other by name without hardcoded IP addresses.

## Sources

- NVIDIA Container Toolkit installation and runtime configuration: [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)
- Docker multi-stage builds: [Docker Documentation — Multi-stage builds](https://docs.docker.com/build/building/multi-stage/)
- Docker Compose service configuration including GPU device reservations: [Docker Documentation — Compose file reference](https://docs.docker.com/compose/compose-file/)
- PyTorch CUDA version compatibility matrix: [PyTorch Documentation — Get Started](https://pytorch.org/get-started/locally/)
- Union filesystem (OverlayFS) and storage driver internals: [Docker Documentation — Storage drivers](https://docs.docker.com/storage/storagedriver/)
- [CITATION NEEDED — concept: GTM Infrastructure cluster §1.4 containerized lead-scoring deployment patterns for outbound enrichment and Salesforce sync]
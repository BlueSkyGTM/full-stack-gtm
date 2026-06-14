## Ship It

Production deployment of containerized AI services introduces three concerns that local development does not exercise: image distribution, GPU scheduling on shared hosts, and version rollback.

**Image distribution.** Your local image exists only on your build machine. To deploy it, you push to a registry — Docker Hub, Amazon ECR, Google Artifact Registry, or a private registry. The registry stores each layer separately and deduplicates across images. When you push a new version that changes only the `COPY app.py` layer (the last layer), the registry transfers only that layer — typically kilobytes, not gigabytes. Tag images with semantic versions, not `latest`:

```bash
docker tag ai-inference your-registry.com/ai-inference:1.2.0
docker tag ai-inference your-registry.com/ai-inference:1.2.0-$(git rev-parse --short HEAD)
docker push your-registry.com/ai-inference:1.2.0
```

**Multi-stage builds for smaller images.** The Dockerfile in Build It uses the `-runtime` CUDA base, which omits compilers. But if you need `flash-attn` — which compiles CUDA kernels at install time — you need a builder stage with the `-devel` base (which includes `nvcc`, `gcc`, and CUDA headers). You compile in the builder, copy the installed wheel, and the final image stays small:

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

RUN apt-get update \
    && apt-get install -y python3.10 python3-pip python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --user --no-cache-dir \
    torch==2.3.0 \
    flash-attn==2.5.8 --no-build-isolation

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update \
    && apt-get install -y python3.10 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/root/.local/lib/python3.10/site-packages

RUN pip install --no-cache-dir fastapi==0.111.0 uvicorn[standard]==0.30.1 pydantic==2.7.1

WORKDIR /app
COPY app.py ./

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Compare image sizes:

```bash
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | grep ai-inference
```

The multi-stage image is typically 1–2 GB smaller than the single-stage equivalent because the final image does not carry `gcc`, `nvcc`, or Python development headers.

**GPU scheduling.** On a host with multiple GPUs, `--gpus all` gives the container access to every GPU. For production, pin specific devices so containers do not contend for the same GPU memory. In Compose, use `count: 1` (lets the runtime pick any free GPU) or specify `device_ids: ["0"]` to pin. For multi-tenant GPU hosts, Kubernetes with the NVIDIA GPU Operator handles scheduling — but that is a separate lesson. The container pattern remains identical: the image is the same, only the orchestrator changes.

**Rollback.** Because each image version is immutable and tagged, rollback is pulling and running the previous tag. If version 1.2.0 produces degraded inference (wrong logits, OOM on longer sequences), `docker run your-registry.com/ai-inference:1.1.4` restores the previous runtime in seconds. This is the same version discipline that applies to outbound infrastructure — if a new enrichment container version produces different lead scores, you roll back to the previous image tag and investigate the diff.
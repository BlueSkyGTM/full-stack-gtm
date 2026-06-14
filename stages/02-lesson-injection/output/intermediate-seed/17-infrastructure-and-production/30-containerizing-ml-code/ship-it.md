## Ship It

### Reducing build context and deploying

Before pushing to a registry, cut down the build context. Docker sends every file in the build directory to the daemon — if your directory contains a 4 GB model checkpoint, a `.venv` folder, or a `data/` directory with training data, every `docker build` transfers all of it over the daemon socket. A `.dockerignore` file excludes these paths.

```text
# save as .dockerignore
.venv/
__pycache__/
*.pyc
data/
models/
*.pkl
*.pt
*.ckpt
.git/
.env
```

This matters for GTM pipelines specifically: if your scoring model weights live in a model registry (S3, MLflow, HuggingFace Hub), they should be downloaded at runtime or pulled in a build stage — never copied into the build context. A 2 GB `.pt` file in your build directory turns every CI build into a multi-minute context transfer. The `.dockerignore` prevents this by default.

Now tag and push to a registry. For GitHub Container Registry:

```bash
docker tag gtm-scorer:v1.3 ghcr.io/your-org/gtm-scorer:v1.3
docker tag gtm-scorer:v1.3 ghcr.io/your-org/gtm-scorer:latest
docker push ghcr.io/your-org/gtm-scorer:v1.3
docker push ghcr.io/your-org/gtm-scorer:latest
```

Two tags serve different purposes. The version tag (`v1.3`) is immutable — it pins the exact image for reproducibility and rollback. The `latest` tag is what your deployment manifest references for automatic pulls. In a GTM scoring pipeline, this means your Clay webhook or Salesforce trigger can always point at `latest` while you retain the ability to audit exactly which model version scored any historical account by looking up which `vX.Y` tag was `latest` at that timestamp.

Verify the deployed image behaves identically by pulling it clean and running the same test:

```bash
docker pull ghcr.io/your-org/gtm-scorer:v1.3
docker run -d -p 8001:8000 --name scorer-verify ghcr.io/your-org/gtm-scorer:v1.3
curl -X POST http://localhost:8001/score \
  -H "Content-Type: application/json" \
  -d '{"employee_count": 6, "revenue_millions": 7}'
```

```bash
docker stop scorer scorer-verify
docker rm scorer scorer-verify
```

The score from the pulled image should match the local build exactly — same numpy, same scikit-learn, same Python. That is the entire promise of containerization: the artifact is the contract.
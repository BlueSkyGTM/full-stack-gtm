# Skill: Containerizing ML Code

## What you built

A multi-stage Dockerfile that packages a FastAPI inference server into a ~150MB production image: builder stage installs dependencies, runtime stage copies only what is needed. Correct layer ordering means code changes rebuild in under 10 seconds.

## Reuse pattern

```dockerfile
# Stage 1: install deps
FROM python:3.12-slim AS builder
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: runtime only
FROM python:3.12-slim AS runtime
COPY --from=builder /install /usr/local
COPY app/ /app/
RUN useradd --uid 1001 --no-create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## .dockerignore template

```
.git/
.venv/
__pycache__/
*.pyc
data/
mlruns/
*.egg-info/
tests/
```

## Next steps

- Push to a registry: `docker tag ml-server ghcr.io/yourname/ml-server:v1 && docker push ...`
- Add a `docker-compose.yml` to run the server alongside a local MLflow tracking server.
- Integrate with Kubernetes (Lesson 31) for orchestrated multi-replica serving.

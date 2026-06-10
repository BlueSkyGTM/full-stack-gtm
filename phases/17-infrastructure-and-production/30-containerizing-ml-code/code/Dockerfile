# Multi-stage Dockerfile — Phase 17/30: Containerizing ML Code
#
# Stage 1 (builder): install Python packages into /install
# Stage 2 (runtime): copy only the installed packages and app code
#
# Layer ordering rationale:
#   requirements.txt changes rarely → copy + pip install cached between code changes
#   app code changes often → copy last so only the final layer misses cache

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Dependencies first — cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Copy only installed packages from builder — no pip, no build tools
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy app code and the saved model
COPY main.py .
COPY model.pkl .

# Non-root user for security best practice
RUN useradd --uid 1001 --no-create-home mluser
USER mluser

EXPOSE 8000

# Single process — no reload in production
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

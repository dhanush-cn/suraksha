# syntax=docker/dockerfile:1.7
#
# Multi-stage build:
#   builder -- has build-essential and the full pip index. Compiles any
#              wheels that need it (bcrypt, cryptography), then produces
#              a self-contained venv at /opt/venv.
#   runtime -- python:3.11-slim only, copies the built venv over, adds
#              a non-root user, and runs alembic + uvicorn.
#
# Why multi-stage:
#   * The runtime image drops build-essential (~250 MB) and gcc, so the
#     shipped surface is smaller and the deploy artifact has fewer CVEs.
#   * A rebuild that changes only Python code (not requirements.txt)
#     re-executes only the runtime stage, so image builds stay fast in CI.
#
# Base image pinned to python:3.11-slim (matches CI + local dev).
# python:3.10-slim was in the old Dockerfile but SQLAlchemy 2.0 async
# features (StrEnum, PEP 695 type params in some deps) prefer 3.11+.

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Compilers only exist in this stage; runtime doesn't need them.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Build a venv so we can bulk-copy it into the runtime image with a
# single COPY --from=builder, keeping layers cache-friendly.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    # These must be overridden at deploy time; no built-in defaults so
    # a misconfigured deploy fails loud on startup.
    JWT_SECRET="" \
    DATABASE_URL="" \
    REDIS_URL="redis://redis:6379/0"

# Runtime-only system deps: libgomp1 for LightGBM/XGBoost/torch's OpenMP
# runtime; curl for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
      curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. UID 10001 sidesteps the common "runs as root" alert
# from image scanners without needing to know the host's UID range.
RUN groupadd --system --gid 10001 rockfall \
    && useradd  --system --uid 10001 --gid rockfall --create-home rockfall

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=rockfall:rockfall backend/  ./backend/
COPY --chown=rockfall:rockfall ml/       ./ml/
COPY --chown=rockfall:rockfall frontend/ ./frontend/

# Model artifacts baked into the image so cold starts don't wait for
# ml/train_model.py to finish. Skips gracefully if the training script
# isn't present -- keeps this Dockerfile buildable even before ml/
# lands.
RUN mkdir -p models && \
    if [ -f ml/train_model.py ]; then python ml/train_model.py; fi && \
    chown -R rockfall:rockfall /app/models

USER rockfall
WORKDIR /app/backend

EXPOSE 8005

# Hit /health/ready (deps included) instead of the aggregated
# /api/health -- Docker's healthcheck should mirror the k8s
# readiness probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:8005/health/ready || exit 1

# Alembic migrations run at container start (idempotent). If the app
# is behind a rolling deployer, migrations still happen exactly once
# because CREATE TABLE IF NOT EXISTS and Alembic's version table
# serialise on the same DB.
CMD ["sh", "-c", "alembic upgrade head && exec python -m uvicorn main:app --host 0.0.0.0 --port 8005"]

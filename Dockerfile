# Production Dockerfile for RockfallGuard Application Stack
FROM python:3.10-slim as base

# Prevent Python from writing bytecode and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install OS build dependencies & OpenMP for XGBoost/LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy & Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Application Code
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY models/ ./models/
COPY frontend/ ./frontend/

# Expose FastAPI Application Port
EXPOSE 8005

# Environment variables
ENV REDIS_HOST=redis \
    REDIS_PORT=6379

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8005/api/health || exit 1

# Launch FastAPI with Uvicorn
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8005"]

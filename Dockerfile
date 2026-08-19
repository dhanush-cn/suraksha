FROM python:3.10-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY ml/ ./ml/
COPY frontend/ ./frontend/

# Generate model artifacts at build time instead of committing them
RUN mkdir -p models && python ml/train_model.py

EXPOSE 8005

ENV REDIS_HOST=redis \
    REDIS_PORT=6379

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8005/api/health || exit 1

# main.py uses sibling imports (from database import ...) which only
# resolve when the working directory is backend/
WORKDIR /app/backend
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
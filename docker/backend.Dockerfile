FROM python:3.11-slim

# Fail fast and keep logs unbuffered so docker logs shows output immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first so the dependency layer is cached between builds.
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

# The store is a volume in compose; create it so a bare `docker run` works too.
RUN mkdir -p /app/chroma_db

EXPOSE 8000

# Ollama runs in its own container, so the default localhost URL is wrong here.
ENV OLLAMA_BASE_URL=http://ollama:11434 \
    CHROMA_PERSIST_DIR=/app/chroma_db

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

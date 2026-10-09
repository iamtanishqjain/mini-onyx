# Mini Onyx

A self-hosted AI platform for document-aware conversations using local LLMs. Built as a full-stack portfolio project demonstrating RAG pipelines, streaming APIs, and containerized deployment.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Backend Tests](https://github.com/iamtanishqjain/mini-onyx/actions/workflows/backend-tests.yml/badge.svg)](https://github.com/iamtanishqjain/mini-onyx/actions/workflows/backend-tests.yml)

---

## Overview

Mini Onyx lets you chat with local LLMs and ask questions from your own documents — entirely on your machine. No API keys, no cloud, no data leaving your system.

The project is inspired by [Onyx](https://github.com/onyx-dot-app/onyx) and built from scratch to explore the internals of a production-grade RAG system: chunking strategies, vector similarity search, streaming inference, and service decomposition.

**Core capabilities:**

- **Retrieval-Augmented Generation (RAG)** — upload PDF, DOCX, or TXT files; the system chunks, embeds, and retrieves relevant context at query time
- **Streaming chat** — responses stream token-by-token over Server-Sent Events
- **Local LLM inference** — powered by [Ollama](https://ollama.ai); runs llama3, mistral, phi3, gemma2 and others
- **Source citations** — every RAG answer exposes which document chunks were used and their similarity scores
- **Model switching** — swap models at runtime from the UI without restarting anything
- **Docker Compose** — single command boots Ollama, pulls models, starts backend and frontend

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │         Next.js Frontend         │
                    │  (TypeScript, Tailwind, SSE)     │
                    └────────────────┬────────────────┘
                                     │ HTTP / SSE
                    ┌────────────────▼────────────────┐
                    │         FastAPI Backend          │
                    │                                  │
                    │  ┌─────────┐   ┌─────────────┐  │
                    │  │  Chat   │   │     RAG     │  │
                    │  │ Router  │   │   Router    │  │
                    │  └────┬────┘   └──────┬──────┘  │
                    │       │               │          │
                    │  ┌────▼────┐   ┌──────▼──────┐  │
                    │  │   LLM   │   │  Embeddings │  │
                    │  │ Service │   │   Service   │  │
                    │  └────┬────┘   └──────┬──────┘  │
                    └───────┼───────────────┼─────────┘
                            │               │
                    ┌───────▼───────────────▼─────────┐
                    │            Ollama               │
                    │   llama3 · nomic-embed-text      │
                    └─────────────────────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │     Vector Store (JSON + cosine) │
                    │         chroma_db/               │
                    └─────────────────────────────────┘
```

**Design decisions:**
- Ollama access is wrapped in a single service (`services/ollama_service.py`) so chat, streaming and embeddings share one client and provider swaps touch one file
- Vector store is implemented in pure Python (cosine similarity over JSON) — no C++ compilation required, making it portable across environments
- System prompts live beside the chat routes, and a prompt supplied by the caller is preserved unless RAG has context to inject
- Streaming uses SSE over a single HTTP connection, emitting `sources`, `token`, `done` and `error` event types

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS | Planned, not in this repository yet |
| Backend | Python 3.11, FastAPI | Async API, SSE streaming, request routing |
| LLM | Ollama (llama3, mistral, phi3) | Local inference, no external API |
| Embeddings | nomic-embed-text via Ollama | Semantic vector generation |
| Vector Store | Pure Python + JSON | Cosine similarity search, zero native deps |
| Document Parsing | pypdf, python-docx | PDF and DOCX text extraction |
| Containerization | Docker Compose | Multi-service orchestration |

---

## Project Structure

```
mini-onyx/
├── docker-compose.yml           # Ollama + backend
├── docker/
│   └── backend.Dockerfile
│
└── backend/
    ├── app/
    │   ├── main.py              # App setup, router registration
    │   ├── core/config.py       # Pydantic settings from .env
    │   └── routers/
    │       ├── chat.py          # POST /chat/stream  POST /chat/
    │       ├── rag.py           # POST /rag/ingest   GET /rag/collections
    │       ├── models.py        # GET /models/
    │       └── health.py        # GET /health
    ├── schemas/schemas.py       # Request and response models
    ├── services/
    │   ├── ollama_service.py    # Chat, streaming, embeddings
    │   └── rag_service.py       # Chunking, ingestion, retrieval
    ├── tests/
    ├── requirements.txt
    └── .env.example
```

The frontend is not in this repository yet; the API is usable through
`/docs` in the meantime.

---

## Getting Started

### Docker (recommended)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```bash
git clone https://github.com/iamtanishqjain/mini-onyx
cd mini-onyx
docker compose up
```

Then pull the models into the running Ollama container, which takes a while the first time (llama3 is ~4 GB):

```bash
docker compose exec ollama ollama pull llama3
docker compose exec ollama ollama pull nomic-embed-text
```

The API is then at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

### Manual Setup

Requires Python 3.11 and [Ollama](https://ollama.ai).

```bash
# Pull required models
ollama pull llama3
ollama pull nomic-embed-text

# Backend
cd backend
py -3.11 -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy .env.example .env
python -m uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/docs`.

---

## API Reference

Interactive docs: `http://localhost:8000/docs`

### `POST /chat/stream`

Streams a chat response as Server-Sent Events.

```json
{
  "messages": [{ "role": "user", "content": "Summarize the document" }],
  "model": "llama3",
  "use_rag": true,
  "collection_name": "my_docs",
  "stream": true
}
```

SSE event types:

| Type | Payload | When |
|------|---------|------|
| `sources` | Array of retrieved chunks with scores | Before generation, if RAG is enabled |
| `token` | Next text token | During generation |
| `done` | — | Stream complete |
| `error` | Error message | On failure |

### `POST /rag/ingest`

Parse, chunk, embed, and store a document.

```bash
curl -X POST http://localhost:8000/rag/ingest \
  -F "file=@report.pdf" \
  -F "collection_name=my_docs"
```

### Other endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Ollama connectivity check |
| `GET` | `/models/` | Available Ollama models |
| `POST` | `/rag/query` | Semantic search within a collection |
| `GET` | `/rag/collections` | List collections with chunk counts |
| `DELETE` | `/rag/collections/{name}` | Remove a collection |

---

## Configuration

```env
# backend/.env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3
OLLAMA_EMBED_MODEL=nomic-embed-text
CHROMA_PERSIST_DIR=./chroma_db
CHUNK_SIZE=500
CHUNK_OVERLAP=50
RAG_TOP_K=5
```

---

## Tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest
```

The suite covers document chunking, collection-name handling, system prompt selection, upload validation and the health endpoint. It runs on every push and pull request against Python 3.10 and 3.12.

---

## Roadmap

- [x] Token streaming over SSE
- [x] RAG pipeline with PDF, DOCX, TXT support
- [x] Cosine similarity vector search
- [x] Source citations with similarity scores
- [x] Runtime model switching
- [x] Centralized prompt management
- [x] Decomposed LLM and embedding services
- [x] Docker Compose orchestration
- [ ] Web search tool (SearXNG)
- [ ] Persistent conversation history
- [ ] MCP tool integrations
- [ ] Cloud deployment guide

---

## License

[MIT](LICENSE) © [Tanishq Jain](https://github.com/iamtanishqjain)

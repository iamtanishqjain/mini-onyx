# Mini Onyx — Backend

FastAPI backend with Ollama (local LLMs), RAG via ChromaDB, and streaming chat.

## Prerequisites

1. **Python 3.10+**
2. **Ollama** — https://ollama.ai
3. Pull required models:
   ```bash
   ollama pull llama3              # chat model
   ollama pull nomic-embed-text    # embedding model
   ```

## Quick Start

```bash
cd backend
chmod +x start.sh
./start.sh
```

Or manually:
```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Check if Ollama is reachable |
| GET | `/models/` | List available Ollama models |
| POST | `/chat/` | Non-streaming chat |
| POST | `/chat/stream` | Streaming chat (SSE) |
| POST | `/rag/ingest` | Upload a document for RAG |
| POST | `/rag/query` | Search documents by query |
| GET | `/rag/collections` | List all document collections |
| DELETE | `/rag/collections/{name}` | Delete a collection |

Interactive docs: **http://localhost:8000/docs**

## Streaming Chat Example

```javascript
const res = await fetch("http://localhost:8000/chat/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    messages: [{ role: "user", content: "What is RAG?" }],
    stream: true
  })
});

const reader = res.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const lines = decoder.decode(value).split("\n");
  for (const line of lines) {
    if (line.startsWith("data: ")) {
      const data = JSON.parse(line.slice(6));
      if (data.type === "token") process.stdout.write(data.content);
      if (data.type === "done") console.log("\n[DONE]");
    }
  }
}
```

## RAG with Chat Example

```bash
# 1. Upload a document
curl -X POST http://localhost:8000/rag/ingest \
  -F "file=@my_document.pdf" \
  -F "collection_name=my_docs"

# 2. Chat with RAG enabled
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Summarize the key points"}],
    "use_rag": true,
    "collection_name": "my_docs",
    "stream": false
  }'
```

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI app + routers
│   ├── core/
│   │   └── config.py        # Settings (from .env)
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response models
│   ├── services/
│   │   ├── ollama_service.py  # Ollama API wrapper (chat + embeddings)
│   │   └── rag_service.py     # Document ingestion + ChromaDB retrieval
│   └── routers/
│       ├── chat.py          # /chat endpoints
│       ├── rag.py           # /rag endpoints
│       ├── models.py        # /models endpoint
│       └── health.py        # /health endpoint
├── requirements.txt
├── .env.example
└── start.sh
```

## Configuration (.env)

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3          # or mistral, phi3, gemma2, etc.
OLLAMA_EMBED_MODEL=nomic-embed-text
CHROMA_PERSIST_DIR=./chroma_db
CHUNK_SIZE=500
CHUNK_OVERLAP=50
RAG_TOP_K=5
```

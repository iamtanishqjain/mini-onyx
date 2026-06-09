# 🚀 Mini Onyx Backend

A production-ready **FastAPI backend** powered by **Ollama**, featuring **local LLM inference**, **Retrieval-Augmented Generation (RAG)**, **document ingestion**, and **real-time streaming chat**.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLMs-orange)
![RAG](https://img.shields.io/badge/RAG-ChromaDB-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ Features

* 🤖 Local AI inference using Ollama
* 💬 Streaming and non-streaming chat APIs
* 📚 Retrieval-Augmented Generation (RAG)
* 📄 Document ingestion and chunking
* 🔍 Semantic search using embeddings
* 🗂️ Collection management
* ⚡ FastAPI-powered REST APIs
* 📖 Interactive Swagger documentation

---

## 🏗️ Tech Stack

| Component      | Technology       |
| -------------- | ---------------- |
| API Framework  | FastAPI          |
| LLM Provider   | Ollama           |
| Vector Storage | ChromaDB         |
| Embeddings     | nomic-embed-text |
| Chat Model     | llama3           |
| Validation     | Pydantic         |
| Server         | Uvicorn          |

---

## 📋 Prerequisites

### 1. Install Python

Python **3.11+** is recommended.

### 2. Install Ollama

Download and install Ollama:

https://ollama.ai

### 3. Pull Required Models

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

Verify:

```bash
ollama list
```

Expected:

```bash
llama3:latest
nomic-embed-text:latest
```

---

## ⚙️ Quick Start

### Clone Repository

```bash
git clone https://github.com/iamtanishqjain/mini-onyx.git
cd mini-onyx/backend
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

```bash
cp .env.example .env
```

### Run Backend

```bash
uvicorn app.main:app --reload --port 8000
```

Backend:

```text
http://localhost:8000
```

Swagger Docs:

```text
http://localhost:8000/docs
```

ReDoc:

```text
http://localhost:8000/redoc
```

---

## 📡 API Endpoints

### System

| Method | Endpoint  | Description             |
| ------ | --------- | ----------------------- |
| GET    | `/health` | Health check            |
| GET    | `/models` | Available Ollama models |

### Chat

| Method | Endpoint       | Description          |
| ------ | -------------- | -------------------- |
| POST   | `/chat`        | Standard chat        |
| POST   | `/chat/stream` | Streaming chat (SSE) |

### RAG

| Method | Endpoint                  | Description       |
| ------ | ------------------------- | ----------------- |
| POST   | `/rag/ingest`             | Upload document   |
| POST   | `/rag/query`              | Query documents   |
| GET    | `/rag/collections`        | List collections  |
| DELETE | `/rag/collections/{name}` | Delete collection |

---

## 💬 Chat Request Example

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is Retrieval Augmented Generation?"
    }
  ],
  "model": "llama3:latest",
  "use_rag": false,
  "stream": false
}
```

---

## 📚 RAG Workflow

### 1. Upload Document

```bash
curl -X POST http://localhost:8000/rag/ingest \
-F "file=@document.pdf" \
-F "collection_name=my_docs"
```

### 2. Query Uploaded Content

```bash
curl -X POST http://localhost:8000/chat \
-H "Content-Type: application/json" \
-d '{
  "messages":[
    {
      "role":"user",
      "content":"Summarize this document"
    }
  ],
  "use_rag":true,
  "collection_name":"my_docs"
}'
```

---

## 🔄 Streaming Chat Example

```javascript
const response = await fetch("http://localhost:8000/chat/stream", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    messages: [
      {
        role: "user",
        content: "Explain vector databases",
      },
    ],
    stream: true,
  }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();

  if (done) break;

  const lines = decoder.decode(value).split("\n");

  for (const line of lines) {
    if (line.startsWith("data: ")) {
      const data = JSON.parse(line.slice(6));

      if (data.type === "token") {
        process.stdout.write(data.content);
      }

      if (data.type === "done") {
        console.log("\n[DONE]");
      }
    }
  }
}
```

---

## 📁 Project Structure

```text
backend/
├── app/
│   ├── main.py
│   ├── core/
│   │   └── config.py
│   └── routers/
│       ├── chat.py
│       ├── rag.py
│       ├── models.py
│       └── health.py
│
├── schemas/
│   └── schemas.py
│
├── services/
│   ├── ollama_service.py
│   └── rag_service.py
│
├── .env.example
├── README.md
├── requirements.txt
└── start.sh
```

---

## 🔧 Environment Variables

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3
OLLAMA_EMBED_MODEL=nomic-embed-text

CHROMA_PERSIST_DIR=./chroma_db

CHUNK_SIZE=500
CHUNK_OVERLAP=50

RAG_TOP_K=5
```

---

## 🛣️ Roadmap

* [ ] Multi-model support
* [ ] Conversation memory
* [ ] User authentication
* [ ] Persistent chat history
* [ ] Hybrid search
* [ ] Docker deployment
* [ ] Kubernetes deployment

---

## 👨‍💻 Author

**Tanishq Jain**

Built as part of the Mini Onyx project using FastAPI, Ollama, and ChromaDB.

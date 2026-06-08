#!/bin/bash
# ── Mini Onyx Backend Startup Script ─────────────────────────────────────────

set -e

echo "🔷 Mini Onyx Backend"
echo "─────────────────────────────────────"

# Check Python
if ! command -v python3 &> /dev/null; then
  echo "❌ Python3 not found. Please install Python 3.10+"
  exit 1
fi

# Setup venv if not exists
if [ ! -d "venv" ]; then
  echo "📦 Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate

# Install deps
echo "📥 Installing dependencies..."
pip install -r requirements.txt -q

# Copy .env if not exists
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "✅ Created .env from .env.example (edit it if needed)"
fi

# Check Ollama
echo ""
echo "🔍 Checking Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "✅ Ollama is running"
else
  echo "⚠️  Ollama not detected at localhost:11434"
  echo "   Install from: https://ollama.ai"
  echo "   Then run:     ollama pull llama3"
  echo "                 ollama pull nomic-embed-text"
  echo ""
fi

# Start server
echo ""
echo "🚀 Starting backend at http://localhost:8000"
echo "📖 API docs at   http://localhost:8000/docs"
echo "─────────────────────────────────────"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

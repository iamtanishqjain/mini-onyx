from pydantic_settings import BaseSettings
from typing import List
import json


class Settings(BaseSettings):
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3"
    ollama_embed_model: str = "nomic-embed-text"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"

    # App
    app_title: str = "Mini Onyx"
    app_version: str = "0.1.0"
    cors_origins: str = '["http://localhost:3000","http://127.0.0.1:3000"]'

    # RAG
    chunk_size: int = 500
    chunk_overlap: int = 50
    rag_top_k: int = 5

    @property
    def cors_origins_list(self) -> List[str]:
        return json.loads(self.cors_origins)

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import json


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3"
    ollama_embed_model: str = "nomic-embed-text"

    # Vector store
    chroma_persist_dir: str = "./chroma_db"

    # App
    app_title: str = "Mini Onyx"
    app_version: str = "0.1.0"
    cors_origins: str = '["http://localhost:3000","http://127.0.0.1:3000"]'

    # Uploads
    max_upload_mb: int = 25

    # RAG
    chunk_size: int = 500
    chunk_overlap: int = 50
    rag_top_k: int = 5

    @property
    def cors_origins_list(self) -> List[str]:
        """Origins as a JSON array or a plain comma-separated list.

        main.py reads this while the app is being imported, so a malformed
        value used to kill startup with a bare JSONDecodeError that named
        neither the setting nor the file it came from.
        """
        raw = self.cors_origins.strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"CORS_ORIGINS is not valid JSON: {e}") from e
        return [origin.strip() for origin in raw.split(",") if origin.strip()]


settings = Settings()

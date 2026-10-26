"""
Centralized configuration using pydantic-settings.

All tunables are loaded from environment variables / .env file.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings with sensible defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Groq LLM ──────────────────────────────────────────
    groq_api_key: str = ""
    llm_model_name: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    # ── Embeddings ────────────────────────────────────────
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # ── ChromaDB ──────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "rag_documents"

    # ── Chunking ──────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ── Retrieval ─────────────────────────────────────────
    retrieval_top_k: int = 5

    # ── API server ────────────────────────────────────────
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    admin_api_key: str = ""

    # ── Logging ───────────────────────────────────────────
    log_level: str = "INFO"

    # ── Derived helpers ───────────────────────────────────
    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir)


# Singleton – import this everywhere
settings = Settings()

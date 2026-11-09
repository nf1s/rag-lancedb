"""Runtime configuration shared by the store, API and UI."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings come from environment variables (case-insensitive) or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # --- store ---
    lancedb_uri: str = "/data/lancedb"
    table_name: str = "documents"
    meta_filename: str = "meta.json"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # --- retrieval ---
    top_k: int = 5
    retrieval_mode: str = "hybrid"
    chunk_size: int = 400
    chunk_overlap: int = 50
    max_context_tokens: int = 3000

    # --- api ---
    store_url: str = "http://localhost:8081"
    store_timeout_seconds: float = 30.0
    store_retries: int = 3
    llm_base_url: str = "http://localhost:1234/v1"
    llm_model: str = "qwen2.5-7b-instruct"
    llm_api_key: str = "lm-studio"
    llm_timeout_seconds: float = 120.0
    llm_health_timeout_seconds: float = 3.0
    embed_batch_size: int = 32

    # --- auth ---
    # Bootstrap credentials are only used when the SQLite user table is empty.
    bootstrap_admin_user: str = "admin"
    bootstrap_admin_password: str = "changeme"
    auth_db_filename: str = "auth.db"
    session_ttl_hours: float = 12.0

    # --- ui ---
    api_url: str = "http://localhost:8080"

    log_level: str = "INFO"

    @property
    def lancedb_path(self) -> Path:
        return Path(self.lancedb_uri)

    @property
    def auth_db_path(self) -> Path:
        """Keep the auth database next to (not inside) the LanceDB directory."""
        return self.lancedb_path.parent / self.auth_db_filename


@lru_cache
def get_settings() -> Settings:
    return Settings()

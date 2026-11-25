from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized, validated configuration loaded from the .env file."""

    # Required API keys (app refuses to start if GROQ_API_KEY is missing)
    groq_api_key: str
    hf_token: str = ""

    # Observability keys (we will activate these in Phase 5)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # LLM Model IDs (stored in one place so model upgrades take 1 line)
    primary_llm_model: str = "openai/gpt-oss-120b"
    fast_llm_model: str = "openai/gpt-oss-20b"
    agent_tool_model: str = "qwen/qwen3.8-27b"

    # Agent Guardrails
    max_agent_steps: int = 6
    llm_temperature: float = 0.0

    # RAG Configuration (Phase 2 - LlamaIndex Hybrid RAG)
    raw_pdf_dir: str = "data/raw_pdfs"
    chroma_persist_dir: str = "chroma_db"
    chroma_collection_name: str = "spectrace_datasheets"
    docstore_path: str = "chroma_db/docstore.json"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings object so .env is read only once."""
    return Settings()
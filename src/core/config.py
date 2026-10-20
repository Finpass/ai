from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEV_INTERNAL_KEY = "development-internal-key"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    port: int = 8001
    log_level: str = "INFO"
    internal_api_key: str = DEV_INTERNAL_KEY
    openrouter_api_key: str = ""
    openrouter_model: str = "anthropic/claude-3.5-haiku"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_timeout_seconds: float = 12
    openrouter_app_url: str = ""
    openrouter_provider_sort: str = ""
    openrouter_provider_order: str = ""
    openrouter_allow_fallbacks: bool = True
    openrouter_data_collection: str = "deny"

    # RAG (optional dependency group `rag`)
    rag_enabled: bool = True
    rag_model: str = ""
    r2_endpoint: str = ""
    r2_bucket: str = "setagama-core"
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    embed_model: str = "BAAI/bge-m3"
    chroma_path: str = "artifacts/chroma_corpus_v2"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rag_top_k: int = 50
    rag_top_n: int = 10
    rag_min_score: float = 0.15
    rag_chat_timeout_seconds: float = 20

    rag_enforce_numbers: bool = False
    rag_llm_titles: bool = True
    rag_max_concurrent: int = 8

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @model_validator(mode="after")
    def _require_real_secret_in_production(self) -> "Settings":
        if self.is_production and self.internal_api_key in ("", DEV_INTERNAL_KEY):
            raise ValueError("INTERNAL_API_KEY must be set to a real secret in production")
        return self


settings = Settings()

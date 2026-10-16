from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEV_INTERNAL_KEY = "development-internal-key"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    port: int = 8001
    log_level: str = "INFO"
    internal_api_key: str = DEV_INTERNAL_KEY

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @model_validator(mode="after")
    def _require_real_secret_in_production(self) -> "Settings":
        if self.is_production and self.internal_api_key in ("", DEV_INTERNAL_KEY):
            raise ValueError("INTERNAL_API_KEY must be set to a real secret in production")
        return self


settings = Settings()

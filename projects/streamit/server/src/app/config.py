"""
Configuration settings using pydantic-settings for environment variable loading.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )

    openai_api_key: str = ""
    openai_ws_url: str = "wss://api.openai.com/v1/realtime"
    translation_model: str = "gpt-5-nano"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    behind_proxy: bool = False
    viewer_tokens: str = ""
    max_seconds_per_user: int | None = 3600

    def get_valid_tokens(self) -> list[str]:
        """Parse comma-separated viewer tokens into a list."""
        return [t.strip() for t in self.viewer_tokens.split(",") if t.strip()]


settings = Settings()

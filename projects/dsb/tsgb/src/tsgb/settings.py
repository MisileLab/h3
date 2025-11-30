"""Configuration management using pydantic-settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Vast.ai API settings (SDK reads from VAST_API_KEY env var by default)
    vast_api_key: str = Field(default="", description="Vast.ai API key")

    # WebDAV / rclone settings for checkpoint storage
    rclone_webdav_url: str | None = Field(
        default=None,
        description="WebDAV URL for remote checkpoint storage",
    )
    rclone_webdav_user: str | None = Field(
        default=None,
        description="WebDAV username",
    )
    rclone_webdav_pass: str | None = Field(
        default=None,
        description="WebDAV password",
    )
    rclone_remote_name: str = Field(
        default="tsgb-remote",
        description="Name for the rclone remote configuration",
    )
    rclone_mountpoint: str = Field(
        default="/mnt/tsgb",
        description="Local mountpoint for WebDAV storage",
    )

    # Checkpoint directories
    checkpoint_dir: str = Field(
        default="/mnt/tsgb/checkpoints",
        description="Directory for storing checkpoints (on mounted storage)",
    )
    local_checkpoint_dir: str = Field(
        default="./checkpoints",
        description="Local checkpoint directory (fallback or local training)",
    )

    # Vast.ai instance filter defaults
    vast_gpu_name: str | None = Field(
        default=None,
        description="GPU model name filter for Vast.ai offers (None = any GPU)",
    )
    vast_min_vram_gb: int = Field(
        default=80,
        description="Minimum VRAM in GB (80GB for H100/A100)",
    )
    vast_max_price: float = Field(
        default=5.0,
        description="Maximum price per hour in USD",
    )

    # Training defaults
    default_model_name: str = Field(
        default="HuggingFaceTB/SmolLM3-3B",
        description="Default model for training (SmolLM3-3B for efficient training)",
    )

    # API keys for black-box LLM providers (Stage 2 evaluation)
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    google_api_key: str = Field(default="", description="Google AI API key")

    # Logging
    log_mode: str = Field(
        default="dev",
        description="Logging mode: 'dev' for colorized console, 'json' for structured JSON",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

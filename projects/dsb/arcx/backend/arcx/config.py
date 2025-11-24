"""Configuration settings for ArcX backend"""

from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field


# Root directory
ROOT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


class CaptureConfig(BaseModel):
    """Screen capture settings"""

    fps: int = Field(default=8, description="Capture frames per second")
    width: int = Field(default=400, description="Downscaled width")
    height: int = Field(default=225, description="Downscaled height")
    buffer_size: int = Field(default=32, description="Ring buffer size (frames)")


class ModelConfig(BaseModel):
    """ML model settings"""

    # Frame encoder
    encoder_dim: int = Field(default=512, description="Latent vector dimension")
    encoder_backbone: Literal["resnet34", "resnet50"] = Field(default="resnet34")

    # Temporal Q-Net
    sequence_length: int = Field(default=32, description="Number of frames in sequence")
    hidden_dim: int = Field(default=512, description="Hidden dimension")
    num_quantiles: int = Field(default=16, description="Number of quantiles for distributional Q")
    temporal_encoder: Literal["gru", "transformer"] = Field(default="gru")

    # Training
    learning_rate: float = Field(default=1e-4)
    batch_size: int = Field(default=32)

    # Model files
    encoder_path: Path = Field(default=MODELS_DIR / "encoder.safetensors")
    qnet_path: Path = Field(default=MODELS_DIR / "qnet.safetensors")


class InferenceConfig(BaseModel):
    """Real-time inference settings"""

    update_interval: float = Field(default=0.5, description="Inference interval in seconds")
    risk_profile: Literal["safe", "neutral", "aggressive"] = Field(default="neutral")

    # Quantile selection based on risk profile
    @property
    def quantile_idx(self) -> float:
        """Get quantile index based on risk profile"""
        if self.risk_profile == "safe":
            return 0.2  # Lower quantile (conservative)
        elif self.risk_profile == "aggressive":
            return 0.8  # Upper quantile (optimistic)
        else:
            return 0.5  # Median


class APIConfig(BaseModel):
    """API server settings"""

    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8765)
    cors_origins: list[str] = Field(default=["http://localhost:*"])


class DataConfig(BaseModel):
    """Data logging settings"""

    log_dir: Path = Field(default=DATA_DIR)
    parquet_compression: Literal["snappy", "gzip", "zstd"] = Field(default="zstd")
    auto_save_interval: int = Field(default=100, description="Save after N decisions")


class Config(BaseModel):
    """Master configuration"""

    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    data: DataConfig = Field(default_factory=DataConfig)


# Global config instance
config = Config()

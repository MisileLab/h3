"""
Configuration management for PDF processing tools.
"""

import os
from pathlib import Path
from typing import final


@final
class Config:
  """Configuration class for PDF processing tools."""
  
  # Default settings
  DEFAULT_MODEL: str = "gpt-4o-mini"
  DEFAULT_TEMPERATURE: float = 0.1
  DEFAULT_BATCH_SIZE: int = 32
  DEFAULT_OUTPUT_FORMAT: str = "csv"
  DEFAULT_OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-pro")
  
  # File extensions
  SUPPORTED_FORMATS: list[str] = ["csv", "txt", "json", "all"]
  
  # Performance settings
  RECOGNITION_BATCH_SIZE: int = int(os.getenv("RECOGNITION_BATCH_SIZE", "32"))
  DETECTOR_BATCH_SIZE: int = int(os.getenv("DETECTOR_BATCH_SIZE", "6"))
  TABLE_REC_BATCH_SIZE: int = int(os.getenv("TABLE_REC_BATCH_SIZE", "8"))
  
  # Device settings
  TORCH_DEVICE: str = os.getenv("TORCH_DEVICE", "cpu")
  
  @classmethod
  def get_openai_api_key(cls) -> str | None:
    """Get OpenAI API key from environment."""
    return os.getenv("OPENAI_API_KEY")
  
  @classmethod
  def validate_openai_key(cls, api_key: str) -> bool:
    """Validate OpenAI API key format."""
    return api_key.startswith("sk-") and len(api_key) > 20
  
  @classmethod
  def validate_openrouter_key(cls, api_key: str) -> bool:
    return len(api_key) > 20
  
  @classmethod
  def get_output_dir(cls, pdf_path: str, output_dir: str | None = None) -> str:
    """Get output directory path."""
    if output_dir:
      return output_dir
    
    pdf_name = Path(pdf_path).stem
    return f"{pdf_name}_extract"
  
  @classmethod
  def get_output_path(cls, pdf_path: str, output_dir: str, format_type: str, 
                     suffix: str = "") -> str:
    """Get output file path."""
    pdf_name = Path(pdf_path).stem
    filename = f"{pdf_name}_{format_type}{suffix}.{format_type}"
    return str(Path(output_dir) / filename) 
"""
Configuration Management Module

Centralized configuration using Pydantic Settings with environment variable support.
Supports development, staging, and production environments.
"""

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ModelConfig(BaseSettings):
    """Model-related configuration."""

    model_config = SettingsConfigDict(env_prefix="MODEL_")

    # Model identifiers
    base_model_name: str = Field(
        default="distilbert-base-uncased",
        description="Hugging Face model identifier for base model",
    )
    fine_tuned_model_path: Path = Field(
        default=Path("models/fine_tuned_model"),
        description="Path to fine-tuned model directory",
    )

    # Training parameters
    max_seq_length: int = Field(default=512, ge=32, le=2048)
    batch_size: int = Field(default=8, ge=1, le=128)
    learning_rate: float = Field(default=2e-5, ge=1e-7, le=1e-2)
    num_epochs: int = Field(default=3, ge=1, le=100)
    warmup_steps: int = Field(default=500, ge=0)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)

    # Inference settings
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=500)

    # Model versioning
    model_version: str = Field(default="v1.0.0")

    @field_validator("fine_tuned_model_path", mode="before")
    @classmethod
    def validate_path(cls, v: str | Path) -> Path:
        """Convert string to Path object."""
        return Path(v) if isinstance(v, str) else v


class APIConfig(BaseSettings):
    """API-related configuration."""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1, le=32)

    # Rate limiting
    rate_limit_requests: int = Field(default=100, ge=1)
    rate_limit_period: int = Field(default=60, ge=1, description="Period in seconds")

    # CORS settings
    cors_origins: list[str] = Field(default=["*"])
    cors_allow_credentials: bool = Field(default=True)
    cors_allow_methods: list[str] = Field(default=["*"])
    cors_allow_headers: list[str] = Field(default=["*"])

    # Request limits
    max_request_size: int = Field(
        default=1_000_000, description="Max request size in bytes"
    )
    request_timeout: int = Field(default=60, description="Request timeout in seconds")


class DataConfig(BaseSettings):
    """Data processing configuration."""

    model_config = SettingsConfigDict(env_prefix="DATA_")

    raw_data_path: Path = Field(default=Path("data/raw"))
    processed_data_path: Path = Field(default=Path("data/processed"))

    # Data splits
    train_split: float = Field(default=0.8, ge=0.1, le=0.99)
    validation_split: float = Field(default=0.1, ge=0.01, le=0.5)
    test_split: float = Field(default=0.1, ge=0.01, le=0.5)

    # Processing settings
    min_text_length: int = Field(default=10, ge=1)
    max_text_length: int = Field(default=10000, ge=100)
    random_seed: int = Field(default=42)

    @field_validator("raw_data_path", "processed_data_path", mode="before")
    @classmethod
    def validate_paths(cls, v: str | Path) -> Path:
        """Convert string to Path object."""
        return Path(v) if isinstance(v, str) else v


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_prefix="LOG_")

    level: str = Field(default="INFO")
    format: str = Field(default="json")
    log_file: Path = Field(default=Path("logs/predictions.log"))
    max_file_size: int = Field(
        default=10_000_000, description="Max log file size in bytes"
    )
    backup_count: int = Field(default=5, ge=1, le=20)

    # Monitoring
    log_predictions: bool = Field(default=True)
    log_latency: bool = Field(default=True)
    log_errors: bool = Field(default=True)

    @field_validator("log_file", mode="before")
    @classmethod
    def validate_log_path(cls, v: str | Path) -> Path:
        """Convert string to Path object."""
        return Path(v) if isinstance(v, str) else v


class Settings(BaseSettings):
    """Main application settings aggregating all configurations."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)
    project_name: str = Field(default="LLM Pipeline")
    project_version: str = Field(default="0.1.0")

    # Base paths
    base_path: Path = Field(default=Path.cwd())

    # Nested configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.data.raw_data_path,
            self.data.processed_data_path,
            self.model.fine_tuned_model_path,
            self.logging.log_file.parent,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses LRU cache to ensure settings are only loaded once.

    Returns:
        Settings: Application settings instance
    """
    return Settings()


# Global settings instance
settings = get_settings()

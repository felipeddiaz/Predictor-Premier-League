"""
Configuration for the PL Predictor API.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class APISettings(BaseSettings):
    """API configuration using environment variables."""

    # Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    API_WORKERS: int = 1

    # CORS
    CORS_ORIGINS: list = ["*"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: list = ["*"]
    CORS_HEADERS: list = ["*"]

    # Predictor
    PREDICTOR_CACHE_SIZE: int = 100
    PREDICTOR_TIMEOUT: int = 30

    # Logging
    LOG_LEVEL: str = "INFO"

    # Feature flags
    ENABLE_GAMEWEEK_PREDICTIONS: bool = True
    ENABLE_TEAM_STATS: bool = True
    ENABLE_HISTORY_TRACKING: bool = False
    ENABLE_UTILS: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = APISettings()

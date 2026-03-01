"""
Configuration settings for RentShield AI Analysis Engine.

Uses Pydantic BaseSettings for environment variable loading with validation.
"""

from functools import lru_cache
from pathlib import Path
from typing import Set

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings can be overridden via environment variables or .env file.
    
    Example:
        >>> settings = get_settings()
        >>> print(settings.OLLAMA_BASE_URL)
        'http://localhost:11434'
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )
    
    # Ollama LLM Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral"
    OLLAMA_TIMEOUT: int = 120
    OLLAMA_MAX_TOKENS: int = 2048
    OLLAMA_TEMPERATURE: float = 0.3
    
    # File Upload Settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png"}
    UPLOAD_DIR: str = "./uploads"
    
    # CORS Settings
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # API Configuration
    API_VERSION: str = "v1"
    API_TITLE: str = "RentShield AI Analysis Engine"
    API_DESCRIPTION: str = (
        "AI-powered evidence verification, issue classification, "
        "and DAO recommendation generation for housing disputes."
    )
    
    # Retry Configuration
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS_ORIGINS string into a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    @property
    def upload_path(self) -> Path:
        """Get upload directory as Path object, creating if needed."""
        path = Path(self.UPLOAD_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    
    Returns:
        Settings: Application settings instance.
        
    Example:
        >>> settings = get_settings()
        >>> settings.OLLAMA_BASE_URL
        'http://localhost:11434'
    """
    return Settings()

"""
Configuration management for ChessTrader.

Handles settings for data sources, cache paths, API limits, and other
system-wide configuration parameters.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import dotenv

# Load environment variables
dotenv.load_dotenv()


@dataclass
class DataSourceConfig:
    """Configuration for data sources and API settings."""

    # Yahoo Finance settings
    yfinance_requests_per_minute: int = 30
    yfinance_retry_attempts: int = 3
    yfinance_retry_delay: float = 1.0

    # Future: Polygon API settings
    polygon_api_key: Optional[str] = None
    polygon_requests_per_minute: int = 5  # Free tier limit

    # Future: Tradier API settings
    tradier_api_key: Optional[str] = None
    tradier_requests_per_minute: int = 120


@dataclass
class CacheConfig:
    """Configuration for caching layer."""

    # Cache database path
    cache_db_path: Path = Path("data/cache.db")

    # Cache TTL settings (in seconds)
    price_data_ttl: int = 24 * 60 * 60  # 24 hours
    options_data_ttl: int = 15 * 60     # 15 minutes
    intraday_data_ttl: int = 5 * 60     # 5 minutes

    # Cache size limits
    max_cache_size_mb: int = 1000  # 1GB
    cache_cleanup_threshold: float = 0.9  # Clean when 90% full


@dataclass
class SystemConfig:
    """System-wide configuration settings."""

    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    # Performance
    max_concurrent_requests: int = 10
    request_timeout: float = 30.0

    # Development/debugging
    debug_mode: bool = False
    enable_profiling: bool = False


@dataclass
class Config:
    """Main configuration container."""

    data_sources: DataSourceConfig = DataSourceConfig()
    cache: CacheConfig = CacheConfig()
    system: SystemConfig = SystemConfig()

    def __post_init__(self):
        """Post-initialization to handle environment variables."""
        # Override with environment variables if available
        self.data_sources.polygon_api_key = os.getenv("POLYGON_API_KEY")
        self.data_sources.tradier_api_key = os.getenv("TRADIER_API_KEY")

        # System config from environment
        if os.getenv("DEBUG"):
            self.system.debug_mode = True

        if os.getenv("LOG_LEVEL"):
            self.system.log_level = os.getenv("LOG_LEVEL")

        # Ensure cache directory exists
        self.cache.cache_db_path.parent.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = Config()
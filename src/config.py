"""
Configuration management for ChessTrader Options AI system.

Handles settings for data sources, cache paths, API limits, recommendations,
backtesting, and other system-wide configuration parameters.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

import dotenv

# Load environment variables
dotenv.load_dotenv()


class DataSourceConfig(BaseModel):
    """Configuration for data sources and API settings."""

    # Yahoo Finance settings
    yfinance_requests_per_minute: int = Field(default=30)
    yfinance_retry_attempts: int = Field(default=3)
    yfinance_retry_delay: float = Field(default=1.0)

    # Future: Polygon API settings
    polygon_api_key: Optional[str] = Field(default=None)
    polygon_requests_per_minute: int = Field(default=5)  # Free tier limit

    # Future: Tradier API settings
    tradier_api_key: Optional[str] = Field(default=None)
    tradier_requests_per_minute: int = Field(default=120)


class CacheConfig(BaseModel):
    """Configuration for caching layer."""

    # Cache database path
    cache_db_path: Path = Field(default=Path("data/cache.db"))

    # Cache TTL settings (in seconds)
    price_data_ttl: int = Field(default=24 * 60 * 60)  # 24 hours
    options_data_ttl: int = Field(default=15 * 60)     # 15 minutes
    intraday_data_ttl: int = Field(default=5 * 60)     # 5 minutes

    # Cache size limits
    max_cache_size_mb: int = Field(default=1000)  # 1GB
    cache_cleanup_threshold: float = Field(default=0.9)  # Clean when 90% full


class TrainingConfig(BaseModel):
    """Training configuration for neural networks."""

    # Core hyperparameters
    learning_rate: float = Field(default=1e-3)
    weight_decay: float = Field(default=1e-4)
    batch_size: int = Field(default=32)
    max_epochs: int = Field(default=100)

    # Early stopping
    patience: int = Field(default=10)
    min_delta: float = Field(default=1e-4)

    # Loss function weights
    loss_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            'classification': 0.8,
            'confidence': 0.2
        }
    )

    # Learning rate scheduler
    lr_scheduler_factor: float = Field(default=0.5)
    lr_scheduler_patience: int = Field(default=5)
    lr_scheduler_min_lr: float = Field(default=1e-6)

    # Training optimizations
    gradient_clip_threshold: float = Field(default=1.0)
    mixed_precision: bool = Field(default=True)

    # Device management
    device: str = Field(default="auto")  # "auto", "cuda", "cpu"

    # Validation and checkpointing
    validation_frequency: int = Field(default=1)  # Validate every N epochs
    checkpoint_frequency: int = Field(default=5)  # Save checkpoint every N epochs
    max_checkpoints: int = Field(default=5)  # Maximum number of checkpoints to keep

    # Paths
    checkpoint_dir: Path = Field(default=Path("models/checkpoints"))
    logs_dir: Path = Field(default=Path("logs/training"))


class RecommendationSettings(BaseModel):
    """Settings for strategy recommendation system."""
    confidence_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for recommendations"
    )
    max_recommendations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of strategy recommendations"
    )
    use_historical_data: bool = Field(
        default=True,
        description="Whether to use historical data for recommendations"
    )


class BacktestingSettings(BaseModel):
    """Settings for backtesting engine."""
    start_date: Optional[str] = Field(
        default=None,
        description="Start date for backtesting (YYYY-MM-DD)"
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date for backtesting (YYYY-MM-DD)"
    )
    initial_capital: float = Field(
        default=100000.0,
        ge=1000.0,
        description="Initial capital for backtesting"
    )
    commission: float = Field(
        default=0.65,
        ge=0.0,
        description="Commission per contract"
    )
    slippage: float = Field(
        default=0.05,
        ge=0.0,
        description="Slippage percentage"
    )
    max_position_size: int = Field(
        default=10,
        ge=1,
        description="Maximum number of contracts per position"
    )
    risk_free_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Risk-free rate for Sharpe ratio calculation"
    )


class APISettings(BaseModel):
    """Settings for API configuration."""
    host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="API server port"
    )
    rate_limit: int = Field(
        default=100,
        ge=1,
        description="API rate limit per minute"
    )
    timeout: int = Field(
        default=30,
        ge=1,
        description="API request timeout in seconds"
    )
    enable_cors: bool = Field(
        default=True,
        description="Enable CORS for API"
    )


class ModelSettings(BaseModel):
    """Settings for model paths and configurations."""
    regime_model_path: Optional[str] = Field(
        default=None,
        description="Path to regime detection model"
    )
    strategy_selector_path: Optional[str] = Field(
        default=None,
        description="Path to strategy selector model"
    )
    position_manager_path: Optional[str] = Field(
        default=None,
        description="Path to position manager model"
    )
    device: str = Field(
        default="cpu",
        description="Device for model inference (cpu/cuda)"
    )


class SystemConfig(BaseModel):
    """System-wide configuration settings."""

    # Logging
    log_level: str = Field(default="INFO")
    log_file: Optional[Path] = Field(default=None)

    # Performance
    max_concurrent_requests: int = Field(default=10)
    request_timeout: float = Field(default=30.0)

    # Development/debugging
    debug_mode: bool = Field(default=False)
    enable_profiling: bool = Field(default=False)


class Config(BaseSettings):
    """Main configuration class for ChessTrader Options AI."""

    model_config = SettingsConfigDict(
        env_prefix="CHESSTRADER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )

    # Sub-configurations
    data_sources: DataSourceConfig = Field(default_factory=DataSourceConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    recommendation: RecommendationSettings = Field(default_factory=RecommendationSettings)
    backtesting: BacktestingSettings = Field(default_factory=BacktestingSettings)
    api: APISettings = Field(default_factory=APISettings)
    models: ModelSettings = Field(default_factory=ModelSettings)

    def __init__(self, **kwargs):
        """Initialize configuration with environment variable overrides."""
        super().__init__(**kwargs)

        # Override with environment variables if available
        self.data_sources.polygon_api_key = os.getenv("POLYGON_API_KEY", self.data_sources.polygon_api_key)
        self.data_sources.tradier_api_key = os.getenv("TRADIER_API_KEY", self.data_sources.tradier_api_key)

        # System config from environment
        if os.getenv("DEBUG"):
            self.system.debug_mode = True

        if os.getenv("LOG_LEVEL"):
            self.system.log_level = os.getenv("LOG_LEVEL")

        # Ensure cache directory exists
        self.cache.cache_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure training directories exist
        self.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.training.logs_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """
        Load configuration from a JSON or Python file.

        Args:
            config_path: Path to configuration file

        Returns:
            Config instance
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if path.suffix == ".json":
            with open(path, "r") as f:
                config_dict = json.load(f)
        elif path.suffix == ".py":
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            config_dict = module.CONFIG if hasattr(module, "CONFIG") else {}
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

        return cls(**config_dict)

    def dict(self, **kwargs) -> Dict[str, Any]:
        """
        Convert configuration to dictionary (Pydantic v1 compatibility).

        Returns:
            Configuration as dictionary
        """
        return self.model_dump(**kwargs)

    def update(self, **kwargs) -> "Config":
        """
        Update configuration with new values.

        Args:
            **kwargs: Configuration values to update

        Returns:
            Updated Config instance
        """
        config_dict = self.dict()

        # Handle nested updates
        for key, value in kwargs.items():
            if "." in key:
                # Handle nested keys like "recommendation.confidence_threshold"
                keys = key.split(".")
                current = config_dict
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                config_dict[key] = value

        return Config(**config_dict)


# Global configuration instance
config = Config()
"""
Unified OptionsAI main class for ChessTrader.

Provides single entry point for strategy recommendations and backtesting
with support for both programmatic API and CLI usage.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json

# Import with fallback for both package and standalone usage
try:
    # Try relative imports first (when used as package)
    from .config import Config
    from .api.strategy_recommender import StrategyRecommender
    from .backtesting.cli.backtest_runner import BacktestCLI
except ImportError:
    # Fallback for standalone usage
    from config import Config
    from api.strategy_recommender import StrategyRecommender
    from backtesting.cli.backtest_runner import BacktestCLI

logger = logging.getLogger(__name__)


class OptionsAI:
    """
    Unified interface for ChessTrader Options AI system.

    Provides clean API for strategy recommendations and backtesting
    with configuration management and error handling.
    """

    def __init__(self, config_path: Optional[str] = None, config: Optional[Config] = None):
        """
        Initialize OptionsAI with configuration.

        Args:
            config_path: Optional path to configuration file
            config: Optional Config instance (takes precedence over config_path)
        """
        # Load configuration
        if config:
            self.config = config
        elif config_path:
            self.config = Config.from_file(config_path)
        else:
            self.config = Config()

        # Setup logging
        self._setup_logging()

        # Initialize components lazily
        self._strategy_recommender: Optional[StrategyRecommender] = None
        self._backtest_cli: Optional[BacktestCLI] = None

        logger.info(f"OptionsAI initialized with config from: {config_path or 'defaults'}")

    def _setup_logging(self):
        """Configure logging based on settings."""
        log_level = getattr(logging, self.config.system.log_level.upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        if self.config.system.log_file:
            file_handler = logging.FileHandler(self.config.system.log_file)
            file_handler.setLevel(log_level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)

    @property
    def strategy_recommender(self) -> StrategyRecommender:
        """Get or create strategy recommender instance."""
        if self._strategy_recommender is None:
            self._strategy_recommender = StrategyRecommender(
                model_path=self.config.models.strategy_selector_path,
                confidence_threshold=self.config.recommendation.confidence_threshold,
                max_recommendations=self.config.recommendation.max_recommendations,
                use_historical_data=self.config.recommendation.use_historical_data
            )
        return self._strategy_recommender

    @property
    def backtest_cli(self) -> BacktestCLI:
        """Get or create backtest CLI instance."""
        if self._backtest_cli is None:
            self._backtest_cli = BacktestCLI()
        return self._backtest_cli

    async def get_recommendations(
        self,
        symbol: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Get strategy recommendations for a symbol.

        Args:
            symbol: Stock/ETF symbol to analyze
            **kwargs: Additional parameters for recommendation engine

        Returns:
            List of strategy recommendations with scores and details

        Example:
            >>> options_ai = OptionsAI()
            >>> recommendations = await options_ai.get_recommendations('SPY')
            >>> for rec in recommendations:
            ...     print(f"{rec['strategy']}: {rec['confidence']:.2%}")
        """
        try:
            # Import data provider
            from .data.providers import get_default_provider

            # Get market data first
            data_provider = get_default_provider()

            # Fetch price history (last 30 days by default)
            price_data = data_provider.get_price_history(symbol)
            current_price = float(price_data.data['Close'].iloc[-1])
            price_history = price_data.data['Close'].values.tolist()

            # Optional: get volume history
            volume_history = price_data.data['Volume'].values.tolist() if 'Volume' in price_data.data.columns else None

            # Run recommendation in executor to avoid blocking
            loop = asyncio.get_event_loop()
            recommendations = await loop.run_in_executor(
                None,
                self.strategy_recommender.recommend,
                symbol,
                current_price,
                price_history,
                volume_history
            )

            logger.info(f"Generated {len(recommendations)} recommendations for {symbol}")
            return recommendations

        except Exception as e:
            logger.error(f"Error getting recommendations for {symbol}: {e}")
            raise

    async def run_backtest(
        self,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run backtest with configuration.

        Args:
            config: Backtest configuration dictionary
            **kwargs: Additional backtest parameters

        Returns:
            Dictionary with backtest results including performance metrics

        Example:
            >>> options_ai = OptionsAI()
            >>> results = await options_ai.run_backtest({
            ...     'symbol': 'SPY',
            ...     'start_date': '2023-01-01',
            ...     'end_date': '2023-12-31',
            ...     'strategy': 'iron_condor'
            ... })
            >>> print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        """
        try:
            # Merge config with defaults from configuration
            backtest_config = {
                'initial_capital': self.config.backtesting.initial_capital,
                'commission': self.config.backtesting.commission,
                'slippage': self.config.backtesting.slippage,
                'max_position_size': self.config.backtesting.max_position_size,
                'risk_free_rate': self.config.backtesting.risk_free_rate
            }

            if self.config.backtesting.start_date:
                backtest_config['start_date'] = self.config.backtesting.start_date
            if self.config.backtesting.end_date:
                backtest_config['end_date'] = self.config.backtesting.end_date

            # Update with provided config
            if config:
                backtest_config.update(config)

            # Update with kwargs
            backtest_config.update(kwargs)

            # Run backtest in executor
            loop = asyncio.get_event_loop()

            # Check if we should use run_complete_workflow or run_backtest
            if hasattr(self.backtest_cli, 'run_complete_workflow'):
                results = await loop.run_in_executor(
                    None,
                    self.backtest_cli.run_complete_workflow,
                    backtest_config
                )
            elif hasattr(self.backtest_cli, 'run_backtest'):
                results = await loop.run_in_executor(
                    None,
                    lambda: self.backtest_cli.run_backtest(**backtest_config)
                )
            else:
                raise AttributeError("BacktestCLI missing required methods")

            logger.info(f"Backtest completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise

    def get_strategy_details(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dictionary with strategy details including description,
            risk profile, and typical use cases

        Example:
            >>> options_ai = OptionsAI()
            >>> details = options_ai.get_strategy_details('iron_condor')
            >>> print(details['description'])
        """
        try:
            return self.strategy_recommender.get_strategy_details(strategy_name)
        except Exception as e:
            logger.error(f"Error getting strategy details for {strategy_name}: {e}")
            raise

    def update_config(self, **kwargs) -> None:
        """
        Update configuration dynamically.

        Args:
            **kwargs: Configuration values to update
                     Supports nested keys with dot notation

        Example:
            >>> options_ai = OptionsAI()
            >>> options_ai.update_config(
            ...     recommendation__confidence_threshold=0.5,
            ...     api__port=8080
            ... )
        """
        # Convert double underscore to dot notation for nested updates
        updates = {}
        for key, value in kwargs.items():
            updates[key.replace('__', '.')] = value

        self.config = self.config.update(**updates)
        logger.info(f"Configuration updated: {updates}")

        # Reset cached components to pick up new config
        self._strategy_recommender = None
        self._backtest_cli = None

    def save_config(self, path: str) -> None:
        """
        Save current configuration to file.

        Args:
            path: Path to save configuration (JSON format)

        Example:
            >>> options_ai = OptionsAI()
            >>> options_ai.save_config('config/my_settings.json')
        """
        config_dict = self.config.dict()

        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            return obj

        config_dict = convert_paths(config_dict)

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Configuration saved to {path}")

    @classmethod
    def version(cls) -> str:
        """
        Get OptionsAI version.

        Returns:
            Version string

        Example:
            >>> print(OptionsAI.version())
            1.0.0
        """
        return "1.0.0"


# Export main class
__all__ = ['OptionsAI']
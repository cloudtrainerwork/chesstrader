"""
Tests for unified OptionsAI main class.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from pathlib import Path
import json
import tempfile

from src.main import OptionsAI
from src.config import Config


class TestOptionsAIInit:
    """Test OptionsAI initialization."""

    def test_options_ai_init_default(self):
        """Test initialization with default configuration."""
        options_ai = OptionsAI()

        assert options_ai is not None
        assert isinstance(options_ai.config, Config)
        assert options_ai.config.recommendation.confidence_threshold == 0.4
        assert options_ai.config.api.port == 8000

    def test_options_ai_init_with_config(self):
        """Test initialization with provided Config instance."""
        config = Config()
        config.recommendation.confidence_threshold = 0.5
        config.api.port = 9000

        options_ai = OptionsAI(config=config)

        assert options_ai.config.recommendation.confidence_threshold == 0.5
        assert options_ai.config.api.port == 9000

    def test_options_ai_init_with_config_path(self):
        """Test initialization with config file path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "recommendation": {
                    "confidence_threshold": 0.6,
                    "max_recommendations": 5
                }
            }
            json.dump(config_data, f)
            config_path = f.name

        try:
            options_ai = OptionsAI(config_path=config_path)
            assert options_ai.config.recommendation.confidence_threshold == 0.6
            assert options_ai.config.recommendation.max_recommendations == 5
        finally:
            Path(config_path).unlink()

    def test_config_loading_invalid_path(self):
        """Test config loading with invalid path."""
        with pytest.raises(FileNotFoundError):
            OptionsAI(config_path="/nonexistent/config.json")


class TestGetRecommendations:
    """Test get_recommendations method."""

    @pytest.mark.asyncio
    async def test_get_recommendations(self):
        """Test getting strategy recommendations."""
        options_ai = OptionsAI()

        # Mock the strategy recommender
        mock_recommender = Mock()
        mock_recommender.recommend.return_value = [
            {"strategy": "iron_condor", "confidence": 0.85, "score": 92},
            {"strategy": "bull_put_spread", "confidence": 0.72, "score": 78}
        ]
        options_ai._strategy_recommender = mock_recommender

        # Get recommendations
        recommendations = await options_ai.get_recommendations("SPY")

        # Verify
        assert len(recommendations) == 2
        assert recommendations[0]["strategy"] == "iron_condor"
        assert recommendations[0]["confidence"] == 0.85
        mock_recommender.recommend.assert_called_once_with("SPY", {})

    @pytest.mark.asyncio
    async def test_get_recommendations_with_kwargs(self):
        """Test getting recommendations with additional parameters."""
        options_ai = OptionsAI()

        # Mock the strategy recommender
        mock_recommender = Mock()
        mock_recommender.recommend.return_value = []
        options_ai._strategy_recommender = mock_recommender

        # Get recommendations with kwargs
        await options_ai.get_recommendations(
            "AAPL",
            time_frame="weekly",
            risk_level="conservative"
        )

        # Verify kwargs were passed
        mock_recommender.recommend.assert_called_once_with(
            "AAPL",
            {"time_frame": "weekly", "risk_level": "conservative"}
        )

    @pytest.mark.asyncio
    async def test_get_recommendations_error_handling(self):
        """Test error handling in get_recommendations."""
        options_ai = OptionsAI()

        # Mock recommender to raise exception
        mock_recommender = Mock()
        mock_recommender.recommend.side_effect = Exception("API error")
        options_ai._strategy_recommender = mock_recommender

        # Should raise exception
        with pytest.raises(Exception) as exc_info:
            await options_ai.get_recommendations("SPY")

        assert "API error" in str(exc_info.value)


class TestRunBacktest:
    """Test run_backtest method."""

    @pytest.mark.asyncio
    async def test_run_backtest_with_complete_workflow(self):
        """Test running backtest with run_complete_workflow method."""
        options_ai = OptionsAI()

        # Mock the backtest CLI
        mock_cli = Mock()
        mock_cli.run_complete_workflow.return_value = {
            "sharpe_ratio": 1.5,
            "total_return": 0.25,
            "max_drawdown": -0.12
        }
        options_ai._backtest_cli = mock_cli

        # Run backtest
        config = {
            "symbol": "SPY",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31"
        }
        results = await options_ai.run_backtest(config)

        # Verify
        assert results["sharpe_ratio"] == 1.5
        assert results["total_return"] == 0.25

        # Check config was merged with defaults
        call_args = mock_cli.run_complete_workflow.call_args[0][0]
        assert call_args["symbol"] == "SPY"
        assert call_args["initial_capital"] == 100000.0
        assert call_args["commission"] == 0.65

    @pytest.mark.asyncio
    async def test_run_backtest_with_run_backtest_method(self):
        """Test running backtest with run_backtest method fallback."""
        options_ai = OptionsAI()

        # Mock the backtest CLI without run_complete_workflow
        mock_cli = Mock(spec=['run_backtest'])
        mock_cli.run_backtest.return_value = {
            "sharpe_ratio": 1.2,
            "total_return": 0.18
        }
        options_ai._backtest_cli = mock_cli

        # Run backtest
        results = await options_ai.run_backtest(symbol="AAPL", strategy="iron_condor")

        # Verify
        assert results["sharpe_ratio"] == 1.2
        mock_cli.run_backtest.assert_called_once()

        # Check kwargs were passed
        call_kwargs = mock_cli.run_backtest.call_args[1]
        assert call_kwargs["symbol"] == "AAPL"
        assert call_kwargs["strategy"] == "iron_condor"

    @pytest.mark.asyncio
    async def test_run_backtest_error_handling(self):
        """Test error handling in run_backtest."""
        options_ai = OptionsAI()

        # Mock CLI to raise exception
        mock_cli = Mock()
        mock_cli.run_complete_workflow.side_effect = ValueError("Invalid date range")
        options_ai._backtest_cli = mock_cli

        # Should raise exception
        with pytest.raises(ValueError) as exc_info:
            await options_ai.run_backtest({})

        assert "Invalid date range" in str(exc_info.value)


class TestStrategyDetails:
    """Test get_strategy_details method."""

    def test_get_strategy_details(self):
        """Test getting strategy details."""
        options_ai = OptionsAI()

        # Mock the strategy recommender
        mock_recommender = Mock()
        mock_recommender.get_strategy_details.return_value = {
            "name": "iron_condor",
            "description": "Neutral strategy with limited risk",
            "risk_profile": "low",
            "market_outlook": "neutral"
        }
        options_ai._strategy_recommender = mock_recommender

        # Get details
        details = options_ai.get_strategy_details("iron_condor")

        # Verify
        assert details["name"] == "iron_condor"
        assert details["risk_profile"] == "low"
        mock_recommender.get_strategy_details.assert_called_once_with("iron_condor")

    def test_get_strategy_details_error_handling(self):
        """Test error handling in get_strategy_details."""
        options_ai = OptionsAI()

        # Mock recommender to raise exception
        mock_recommender = Mock()
        mock_recommender.get_strategy_details.side_effect = KeyError("Unknown strategy")
        options_ai._strategy_recommender = mock_recommender

        # Should raise exception
        with pytest.raises(KeyError) as exc_info:
            options_ai.get_strategy_details("invalid_strategy")

        assert "Unknown strategy" in str(exc_info.value)


class TestConfigManagement:
    """Test configuration management methods."""

    def test_update_config(self):
        """Test updating configuration dynamically."""
        options_ai = OptionsAI()
        original_threshold = options_ai.config.recommendation.confidence_threshold

        # Update config with nested keys
        options_ai.update_config(
            recommendation__confidence_threshold=0.7,
            api__port=9090
        )

        # Verify updates
        assert options_ai.config.recommendation.confidence_threshold == 0.7
        assert options_ai.config.api.port == 9090

        # Verify cached components are reset
        assert options_ai._strategy_recommender is None
        assert options_ai._backtest_cli is None

    def test_save_config(self):
        """Test saving configuration to file."""
        options_ai = OptionsAI()
        options_ai.config.recommendation.confidence_threshold = 0.8

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            # Save config
            options_ai.save_config(config_path)

            # Load and verify
            with open(config_path, 'r') as f:
                saved_config = json.load(f)

            assert saved_config["recommendation"]["confidence_threshold"] == 0.8
            assert saved_config["api"]["port"] == 8000
        finally:
            Path(config_path).unlink()


class TestLazyLoading:
    """Test lazy loading of components."""

    def test_strategy_recommender_lazy_load(self):
        """Test lazy loading of strategy recommender."""
        options_ai = OptionsAI()

        # Initially not loaded
        assert options_ai._strategy_recommender is None

        # Access property triggers loading
        with patch('src.main.StrategyRecommender') as MockRecommender:
            recommender = options_ai.strategy_recommender

            # Verify it was created with correct config
            MockRecommender.assert_called_once_with(
                model_path=None,
                confidence_threshold=0.4,
                max_recommendations=3,
                use_historical_data=True
            )

            # Subsequent access returns same instance
            recommender2 = options_ai.strategy_recommender
            assert recommender is recommender2

    def test_backtest_cli_lazy_load(self):
        """Test lazy loading of backtest CLI."""
        options_ai = OptionsAI()

        # Initially not loaded
        assert options_ai._backtest_cli is None

        # Access property triggers loading
        with patch('src.main.BacktestCLI') as MockCLI:
            cli = options_ai.backtest_cli

            # Verify it was created
            MockCLI.assert_called_once()

            # Subsequent access returns same instance
            cli2 = options_ai.backtest_cli
            assert cli is cli2


class TestVersion:
    """Test version method."""

    def test_version(self):
        """Test getting version string."""
        version = OptionsAI.version()
        assert version == "1.0.0"
        assert isinstance(version, str)


def test_options_ai_init():
    """Simple integration test for OptionsAI initialization."""
    # This test is explicitly required by the plan
    options_ai = OptionsAI()
    assert options_ai is not None
    assert hasattr(options_ai, 'get_recommendations')
    assert hasattr(options_ai, 'run_backtest')
    assert hasattr(options_ai, 'get_strategy_details')


def test_get_recommendations():
    """Simple test for get_recommendations method delegation."""
    # This test is explicitly required by the plan
    options_ai = OptionsAI()

    # Mock the strategy recommender
    mock_recommender = Mock()
    mock_recommender.recommend.return_value = [
        {"strategy": "test_strategy", "confidence": 0.9}
    ]
    options_ai._strategy_recommender = mock_recommender

    # Run async method synchronously for simple test
    loop = asyncio.new_event_loop()
    recommendations = loop.run_until_complete(
        options_ai.get_recommendations("TEST")
    )
    loop.close()

    # Verify delegation
    assert len(recommendations) == 1
    assert recommendations[0]["strategy"] == "test_strategy"
    mock_recommender.recommend.assert_called_once()
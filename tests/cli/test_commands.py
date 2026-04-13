"""
Tests for CLI commands.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typer.testing import CliRunner

from src.cli.main import app


class TestRecommendCommand:
    """Test recommend command functionality."""

    def setup_method(self):
        """Setup for each test method."""
        self.runner = CliRunner()

    @patch('src.cli.commands.recommend.get_options_ai')
    @patch('src.cli.commands.recommend.run_async')
    def test_recommend_basic(self, mock_run_async, mock_get_ai):
        """Test basic recommend command."""
        # Mock OptionsAI
        mock_ai = Mock()
        mock_get_ai.return_value = mock_ai

        # Mock recommendations
        mock_recommendations = [
            {
                'strategy': 'iron_condor',
                'confidence': 0.85,
                'score': 92,
                'market_outlook': 'neutral'
            },
            {
                'strategy': 'bull_put_spread',
                'confidence': 0.72,
                'score': 78,
                'market_outlook': 'bullish'
            }
        ]
        mock_run_async.return_value = mock_recommendations

        # Run command
        result = self.runner.invoke(app, ["recommend", "AAPL"])

        # Verify
        assert result.exit_code == 0
        assert "Strategy Recommendations for AAPL" in result.stdout
        assert "Iron Condor" in result.stdout
        assert "Bull Put Spread" in result.stdout
        assert "85.0%" in result.stdout

    @patch('src.cli.commands.recommend.get_options_ai')
    @patch('src.cli.commands.recommend.run_async')
    def test_recommend_with_options(self, mock_run_async, mock_get_ai):
        """Test recommend command with options."""
        # Mock OptionsAI
        mock_ai = Mock()
        mock_get_ai.return_value = mock_ai
        mock_ai.update_config = Mock()

        # Mock recommendations
        mock_recommendations = [
            {
                'strategy': 'covered_call',
                'confidence': 0.65,
                'score': 75,
                'market_outlook': 'slightly_bullish'
            }
        ]
        mock_run_async.return_value = mock_recommendations

        # Run command with options
        result = self.runner.invoke(app, [
            "recommend", "SPY",
            "--confidence", "0.6",
            "--max-results", "5"
        ])

        # Verify
        assert result.exit_code == 0
        mock_ai.update_config.assert_called_once_with(
            recommendation__confidence_threshold=0.6,
            recommendation__max_recommendations=5
        )

    @patch('src.cli.commands.recommend.get_options_ai')
    @patch('src.cli.commands.recommend.run_async')
    def test_recommend_with_details(self, mock_run_async, mock_get_ai):
        """Test recommend command with details flag."""
        # Mock OptionsAI
        mock_ai = Mock()
        mock_get_ai.return_value = mock_ai
        mock_ai.get_strategy_details.return_value = {
            'description': 'Low-risk neutral strategy',
            'risk_profile': 'low',
            'typical_duration': '30-45 days'
        }

        # Mock recommendations
        mock_recommendations = [
            {
                'strategy': 'iron_butterfly',
                'confidence': 0.88,
                'score': 95,
                'market_outlook': 'neutral'
            }
        ]
        mock_run_async.return_value = mock_recommendations

        # Run command with details
        result = self.runner.invoke(app, ["recommend", "QQQ", "--details"])

        # Verify
        assert result.exit_code == 0
        assert "Risk Level" in result.stdout
        assert "Time Frame" in result.stdout
        assert "Strategy Details" in result.stdout

    @patch('src.cli.commands.recommend.get_options_ai')
    @patch('src.cli.commands.recommend.run_async')
    def test_recommend_no_results(self, mock_run_async, mock_get_ai):
        """Test recommend command with no results."""
        # Mock OptionsAI
        mock_ai = Mock()
        mock_get_ai.return_value = mock_ai

        # Mock empty recommendations
        mock_run_async.return_value = []

        # Run command
        result = self.runner.invoke(app, ["recommend", "INVALID"])

        # Verify
        assert result.exit_code == 0
        assert "No recommendations found" in result.stdout
        assert "Confidence threshold too high" in result.stdout

    @patch('src.cli.commands.recommend.get_options_ai')
    @patch('src.cli.commands.recommend.run_async')
    def test_recommend_error_handling(self, mock_run_async, mock_get_ai):
        """Test recommend command error handling."""
        # Mock OptionsAI
        mock_ai = Mock()
        mock_get_ai.return_value = mock_ai

        # Mock exception
        mock_run_async.side_effect = Exception("API error")

        # Run command
        result = self.runner.invoke(app, ["recommend", "AAPL"])

        # Verify error handling
        assert result.exit_code == 1
        assert "Error getting recommendations" in result.stdout

    def test_recommend_help(self):
        """Test recommend command help."""
        result = self.runner.invoke(app, ["recommend", "--help"])

        assert result.exit_code == 0
        assert "strategy recommendations" in result.stdout.lower()
        assert "--confidence" in result.stdout
        assert "--max-results" in result.stdout
        assert "--details" in result.stdout


class TestBacktestCommand:
    """Test backtest command functionality."""

    def setup_method(self):
        """Setup for each test method."""
        self.runner = CliRunner()

    @patch('src.cli.commands.backtest.get_options_ai')
    @patch('src.cli.commands.backtest.run_async')
    def test_backtest_basic(self, mock_run_async, mock_get_ai):
        """Test basic backtest command."""
        # Mock OptionsAI
        mock_ai = Mock()
        mock_get_ai.return_value = mock_ai

        # Mock backtest results
        mock_results = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'win_rate': 0.65,
            'total_trades': 24,
            'avg_trade_return': 0.02,
            'volatility': 0.12,
            'sortino_ratio': 1.5,
            'calmar_ratio': 2.1,
            'var_95': -0.05
        }
        mock_run_async.return_value = mock_results

        # Run command
        result = self.runner.invoke(app, ["backtest", "--symbol", "AAPL"])

        # Verify
        assert result.exit_code == 0
        assert "Backtesting Analysis for AAPL" in result.stdout
        assert "Key Performance Metrics" in result.stdout
        assert "15.00%" in result.stdout  # Total return
        assert "1.20" in result.stdout     # Sharpe ratio

    @patch('src.cli.commands.backtest.get_options_ai')
    @patch('src.cli.commands.backtest.run_async')
    def test_backtest_with_strategy(self, mock_run_async, mock_get_ai):
        """Test backtest command with specific strategy."""
        # Mock OptionsAI
        mock_ai = Mock()
        mock_get_ai.return_value = mock_ai

        # Mock backtest results
        mock_results = {
            'total_return': 0.22,
            'sharpe_ratio': 1.8,
            'max_drawdown': -0.12,
            'win_rate': 0.72,
            'total_trades': 18,
            'avg_trade_return': 0.03
        }
        mock_run_async.return_value = mock_results

        # Run command with strategy
        result = self.runner.invoke(app, [
            "backtest",
            "--symbol", "SPY",
            "--strategy", "iron_condor",
            "--start-date", "2023-01-01",
            "--end-date", "2023-12-31"
        ])

        # Verify
        assert result.exit_code == 0
        assert "Strategy: Iron Condor" in result.stdout
        assert "2023-01-01" in result.stdout

    @patch('src.cli.commands.backtest.get_options_ai')
    @patch('src.cli.commands.backtest.run_async')
    def test_backtest_with_options(self, mock_run_async, mock_get_ai):
        """Test backtest command with custom options."""
        # Mock OptionsAI
        mock_ai = Mock()
        mock_get_ai.return_value = mock_ai

        # Mock backtest results
        mock_results = {
            'total_return': 0.08,
            'sharpe_ratio': 0.9,
            'max_drawdown': -0.15,
            'win_rate': 0.58,
            'total_trades': 32,
            'avg_trade_return': 0.015
        }
        mock_run_async.return_value = mock_results

        # Run command with custom options
        result = self.runner.invoke(app, [
            "backtest",
            "--symbol", "QQQ",
            "--capital", "50000",
            "--commission", "1.0",
            "--max-positions", "5"
        ])

        # Verify
        assert result.exit_code == 0
        assert "Capital: $50,000" in result.stdout

    @patch('src.cli.commands.backtest.get_options_ai')
    @patch('src.cli.commands.backtest.run_async')
    def test_backtest_with_trade_stats(self, mock_run_async, mock_get_ai):
        """Test backtest command with detailed trade statistics."""
        # Mock OptionsAI
        mock_ai = Mock()
        mock_get_ai.return_value = mock_ai

        # Mock backtest results with trade stats
        mock_results = {
            'total_return': 0.18,
            'sharpe_ratio': 1.4,
            'max_drawdown': -0.09,
            'win_rate': 0.68,
            'total_trades': 28,
            'avg_trade_return': 0.025,
            'trade_stats': {
                'profitable_trades': 19,
                'loss_trades': 9,
                'largest_win': 0.08,
                'largest_loss': -0.04,
                'avg_win': 0.045,
                'avg_loss': -0.022
            }
        }
        mock_run_async.return_value = mock_results

        # Run command
        result = self.runner.invoke(app, ["backtest", "--symbol", "MSFT"])

        # Verify
        assert result.exit_code == 0
        assert "Trade Statistics" in result.stdout
        assert "Profitable Trades" in result.stdout
        assert "19" in result.stdout  # profitable trades count

    @patch('src.cli.commands.backtest.get_options_ai')
    @patch('src.cli.commands.backtest.run_async')
    def test_backtest_no_results(self, mock_run_async, mock_get_ai):
        """Test backtest command with no results."""
        # Mock OptionsAI
        mock_ai = Mock()
        mock_get_ai.return_value = mock_ai

        # Mock empty results
        mock_run_async.return_value = None

        # Run command
        result = self.runner.invoke(app, ["backtest", "--symbol", "INVALID"])

        # Verify
        assert result.exit_code == 0
        assert "No backtest results generated" in result.stdout
        assert "Insufficient historical data" in result.stdout

    @patch('src.cli.commands.backtest.get_options_ai')
    @patch('src.cli.commands.backtest.run_async')
    def test_backtest_save_report(self, mock_run_async, mock_get_ai):
        """Test backtest command with save report option."""
        # Mock OptionsAI
        mock_ai = Mock()
        mock_get_ai.return_value = mock_ai

        # Mock backtest results
        mock_results = {
            'total_return': 0.12,
            'sharpe_ratio': 1.1,
            'max_drawdown': -0.07,
            'win_rate': 0.62
        }
        mock_run_async.return_value = mock_results

        # Run command with save report
        result = self.runner.invoke(app, [
            "backtest",
            "--symbol", "AMZN",
            "--save-report",
            "--output", "html"
        ])

        # Verify
        assert result.exit_code == 0
        assert "Report saved" in result.stdout
        assert "html" in result.stdout.lower()

    @patch('src.cli.commands.backtest.get_options_ai')
    @patch('src.cli.commands.backtest.run_async')
    def test_backtest_error_handling(self, mock_run_async, mock_get_ai):
        """Test backtest command error handling."""
        # Mock OptionsAI
        mock_ai = Mock()
        mock_get_ai.return_value = mock_ai

        # Mock exception
        mock_run_async.side_effect = Exception("Backtest error")

        # Run command
        result = self.runner.invoke(app, ["backtest", "--symbol", "AAPL"])

        # Verify error handling
        assert result.exit_code == 1
        assert "Error running backtest" in result.stdout

    def test_backtest_help(self):
        """Test backtest command help."""
        result = self.runner.invoke(app, ["backtest", "--help"])

        assert result.exit_code == 0
        assert "backtesting" in result.stdout.lower()
        assert "--symbol" in result.stdout
        assert "--strategy" in result.stdout
        assert "--start-date" in result.stdout
        assert "--capital" in result.stdout


class TestCommandIntegration:
    """Test command integration and error scenarios."""

    def setup_method(self):
        """Setup for each test method."""
        self.runner = CliRunner()

    def test_invalid_command(self):
        """Test invalid command handling."""
        result = self.runner.invoke(app, ["invalid_command"])
        assert result.exit_code != 0

    def test_missing_required_args(self):
        """Test missing required arguments."""
        # Recommend command requires symbol
        result = self.runner.invoke(app, ["recommend"])
        assert result.exit_code != 0

        # Backtest command requires symbol
        result = self.runner.invoke(app, ["backtest"])
        assert result.exit_code != 0
"""
Tests for CLI main application.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner

from src.cli.main import app, get_options_ai


class TestCLIMain:
    """Test main CLI application functionality."""

    def setup_method(self):
        """Setup for each test method."""
        self.runner = CliRunner()

    def test_app_help(self):
        """Test CLI help output."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "ChessTrader Options AI" in result.stdout
        assert "recommend" in result.stdout
        assert "backtest" in result.stdout

    def test_version_flag(self):
        """Test version flag displays version info."""
        with patch('src.cli.main.get_options_ai') as mock_get_ai:
            mock_ai = Mock()
            mock_ai.version.return_value = "1.0.0"
            mock_get_ai.return_value = mock_ai

            result = self.runner.invoke(app, ["--version"])
            assert result.exit_code == 0
            assert "1.0.0" in result.stdout

    def test_version_command(self):
        """Test version command displays detailed info."""
        with patch('src.cli.main.get_options_ai') as mock_get_ai:
            mock_ai = Mock()
            mock_ai.version.return_value = "1.0.0"
            mock_get_ai.return_value = mock_ai

            result = self.runner.invoke(app, ["version"])
            assert result.exit_code == 0
            assert "ChessTrader Options AI" in result.stdout
            assert "Chess-inspired neural architecture" in result.stdout

    def test_config_option(self):
        """Test config option is accepted."""
        with patch('src.cli.main.get_options_ai') as mock_get_ai:
            mock_ai = Mock()
            mock_ai.version.return_value = "1.0.0"
            mock_get_ai.return_value = mock_ai

            # Test with non-existent config file - should show file not found
            result = self.runner.invoke(app, ["--config", "/nonexistent/config.json", "--version"])
            # Should fail due to file validation
            assert result.exit_code != 0

    def test_verbose_option(self):
        """Test verbose option is accepted."""
        with patch('src.cli.main.get_options_ai') as mock_get_ai:
            mock_ai = Mock()
            mock_ai.version.return_value = "1.0.0"
            mock_get_ai.return_value = mock_ai

            result = self.runner.invoke(app, ["--verbose", "--version"])
            assert result.exit_code == 0
            assert "Verbose mode enabled" in result.stdout

    def test_get_options_ai_caching(self):
        """Test that OptionsAI instance is cached."""
        # Reset global state
        import src.cli.main
        src.cli.main._options_ai = None

        with patch('src.cli.main.OptionsAI') as MockOptionsAI:
            mock_instance = Mock()
            MockOptionsAI.return_value = mock_instance

            # First call should create instance
            ai1 = get_options_ai()
            assert ai1 is mock_instance
            MockOptionsAI.assert_called_once_with(config_path=None)

            # Second call should return cached instance
            ai2 = get_options_ai()
            assert ai2 is mock_instance
            assert ai1 is ai2
            # Should not call constructor again
            assert MockOptionsAI.call_count == 1

    def test_get_options_ai_error_handling(self):
        """Test error handling in get_options_ai."""
        # Reset global state
        import src.cli.main
        src.cli.main._options_ai = None

        with patch('src.cli.main.OptionsAI') as MockOptionsAI:
            MockOptionsAI.side_effect = Exception("Config error")

            with pytest.raises(SystemExit):
                get_options_ai()


class TestAsyncHelpers:
    """Test async helper functions."""

    def test_run_async_simple(self):
        """Test run_async with simple coroutine."""
        from src.cli.main import run_async
        import asyncio

        async def simple_coro():
            return "test_result"

        result = run_async(simple_coro())
        assert result == "test_result"

    def test_run_async_with_exception(self):
        """Test run_async handles exceptions."""
        from src.cli.main import run_async
        import asyncio

        async def error_coro():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            run_async(error_coro())


class TestCommandRegistration:
    """Test that commands are properly registered."""

    def setup_method(self):
        """Setup for each test method."""
        self.runner = CliRunner()

    def test_recommend_command_registered(self):
        """Test recommend command is registered."""
        result = self.runner.invoke(app, ["recommend", "--help"])
        assert result.exit_code == 0
        assert "strategy recommendations" in result.stdout.lower()

    def test_backtest_command_registered(self):
        """Test backtest command is registered."""
        result = self.runner.invoke(app, ["backtest", "--help"])
        assert result.exit_code == 0
        assert "backtesting" in result.stdout.lower()

    def test_command_list(self):
        """Test that all expected commands are available."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0

        # Check main commands are listed
        assert "recommend" in result.stdout
        assert "backtest" in result.stdout
        assert "version" in result.stdout
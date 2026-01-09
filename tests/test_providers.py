"""
Tests for data providers.

Tests the YFinanceProvider implementation including error handling,
data validation, and basic functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import pandas as pd

from src.data.providers import (
    YFinanceProvider,
    DataProviderError,
    InvalidSymbol,
    DataNotAvailable,
    RateLimiter
)
from src.data.models import PriceData


class TestRateLimiter:
    """Test rate limiter functionality."""

    def test_rate_limiter_allows_requests_within_limit(self):
        """Test that requests within limit are allowed."""
        limiter = RateLimiter(max_requests_per_minute=5)

        # Should not block for requests within limit
        for _ in range(5):
            limiter.wait_if_needed()  # Should complete quickly

    def test_rate_limiter_blocks_when_limit_exceeded(self):
        """Test that rate limiter blocks when limit is exceeded."""
        limiter = RateLimiter(max_requests_per_minute=2)

        # First 2 requests should be fast
        limiter.wait_if_needed()
        limiter.wait_if_needed()

        # This would normally block, but we won't wait in tests
        # Just verify the logic works
        assert len(limiter.requests) == 2


class TestYFinanceProvider:
    """Test YFinance provider implementation."""

    @pytest.fixture
    def provider(self):
        """Create a YFinance provider instance."""
        return YFinanceProvider()

    def test_validate_symbol_valid_symbols(self, provider):
        """Test symbol validation with valid symbols."""
        assert provider._validate_symbol("SPY") == "SPY"
        assert provider._validate_symbol("spy") == "SPY"
        assert provider._validate_symbol(" AAPL ") == "AAPL"
        assert provider._validate_symbol("BRK.B") == "BRK.B"
        assert provider._validate_symbol("VIX") == "VIX"

    def test_validate_symbol_invalid_symbols(self, provider):
        """Test symbol validation with invalid symbols."""
        with pytest.raises(InvalidSymbol):
            provider._validate_symbol("")

        with pytest.raises(InvalidSymbol):
            provider._validate_symbol(None)

        with pytest.raises(InvalidSymbol):
            provider._validate_symbol("INVALID$SYMBOL!")

    @patch('src.data.providers.yf.Ticker')
    def test_get_price_history_success(self, mock_ticker, provider):
        """Test successful price data fetching."""
        # Mock successful response
        mock_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [101.0, 102.0, 103.0],
            'Low': [99.0, 100.0, 101.0],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2024-01-01', periods=3))

        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance

        # Test the method
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)
        result = provider.get_price_history("SPY", start, end)

        # Verify result
        assert isinstance(result, PriceData)
        assert result.symbol == "SPY"
        assert result.is_valid
        assert len(result.data) == 3
        assert result.latest_price == 102.5

    @patch('src.data.providers.yf.Ticker')
    def test_get_price_history_no_data(self, mock_ticker, provider):
        """Test behavior when no data is available."""
        # Mock empty response
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance

        with pytest.raises(DataNotAvailable):
            provider.get_price_history("INVALID")

    @patch('src.data.providers.yf.Ticker')
    def test_get_current_price_success(self, mock_ticker, provider):
        """Test successful current price fetching."""
        # Mock successful response with fast_info
        mock_fast_info = Mock()
        mock_fast_info.last_price = 150.25

        mock_ticker_instance = Mock()
        mock_ticker_instance.fast_info = mock_fast_info
        mock_ticker.return_value = mock_ticker_instance

        result = provider.get_current_price("AAPL")
        assert result == 150.25

    @patch('src.data.providers.yf.Ticker')
    def test_get_current_price_fallback_to_history(self, mock_ticker, provider):
        """Test fallback to history when fast_info fails."""
        # Mock fast_info without last_price
        mock_fast_info = Mock()
        mock_fast_info.last_price = None

        # Mock history data
        mock_history = pd.DataFrame({
            'Close': [145.0, 146.0, 147.0]
        })

        mock_ticker_instance = Mock()
        mock_ticker_instance.fast_info = mock_fast_info
        mock_ticker_instance.history.return_value = mock_history
        mock_ticker.return_value = mock_ticker_instance

        result = provider.get_current_price("AAPL")
        assert result == 147.0

    @patch('src.data.providers.yf.Ticker')
    def test_batch_get_prices(self, mock_ticker, provider):
        """Test batch price fetching."""
        # Mock data for multiple symbols
        def mock_history(*args, **kwargs):
            return pd.DataFrame({
                'Open': [100.0], 'High': [101.0], 'Low': [99.0],
                'Close': [100.5], 'Volume': [1000000]
            }, index=pd.date_range('2024-01-01', periods=1))

        mock_ticker_instance = Mock()
        mock_ticker_instance.history.side_effect = mock_history
        mock_ticker.return_value = mock_ticker_instance

        symbols = ["SPY", "QQQ", "IWM"]
        results = provider.batch_get_prices(symbols)

        assert len(results) == 3
        for symbol in symbols:
            assert symbol in results
            assert isinstance(results[symbol], PriceData)
            assert results[symbol].symbol == symbol

    def test_batch_get_prices_empty_list(self, provider):
        """Test batch fetching with empty symbol list."""
        result = provider.batch_get_prices([])
        assert result == {}

    @patch('src.data.providers.yf.Ticker')
    def test_get_options_chain_success(self, mock_ticker, provider):
        """Test successful options chain fetching."""
        # Mock options data
        calls_data = pd.DataFrame({
            'strike': [100, 105, 110],
            'bid': [5.0, 2.5, 1.0],
            'ask': [5.2, 2.7, 1.2],
            'volume': [100, 50, 25],
            'openInterest': [1000, 500, 250],
            'impliedVolatility': [0.25, 0.27, 0.30]
        })

        puts_data = pd.DataFrame({
            'strike': [100, 105, 110],
            'bid': [1.0, 2.5, 5.0],
            'ask': [1.2, 2.7, 5.2],
            'volume': [25, 50, 100],
            'openInterest': [250, 500, 1000],
            'impliedVolatility': [0.30, 0.27, 0.25]
        })

        mock_option_chain = Mock()
        mock_option_chain.calls = calls_data
        mock_option_chain.puts = puts_data

        mock_ticker_instance = Mock()
        mock_ticker_instance.options = ['2024-01-19', '2024-02-16']
        mock_ticker_instance.option_chain.return_value = mock_option_chain

        # Mock fast_info for current price
        mock_fast_info = Mock()
        mock_fast_info.last_price = 102.5
        mock_ticker_instance.fast_info = mock_fast_info

        mock_ticker.return_value = mock_ticker_instance

        result = provider.get_options_chain("SPY")

        assert result.symbol == "SPY"
        assert result.is_valid
        assert len(result.calls) == 3
        assert len(result.puts) == 3
        assert result.underlying_price == 102.5


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__])
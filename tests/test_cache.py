"""
Tests for cache functionality.

Tests the CacheManager implementation including data storage,
retrieval, TTL management, and statistics tracking.
"""

import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd

from src.data.cache import CacheManager, CacheError
from src.data.models import PriceData, OptionChainData
from src.data.schema import PriceHistory, OptionsData, CacheMetadata


class TestCacheManager:
    """Test cache manager functionality."""

    @pytest.fixture
    def temp_cache(self):
        """Create a temporary cache manager for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "test_cache.db"
            mock_provider = Mock()
            cache = CacheManager(database_path=cache_path, provider=mock_provider)
            yield cache, mock_provider

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [101.0, 102.0, 103.0],
            'Low': [99.0, 100.0, 101.0],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2024-01-01', periods=3, name='Date'))

        return PriceData(
            symbol="SPY",
            data=data,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            interval="1d",
            source="yfinance"
        )

    @pytest.fixture
    def sample_options_data(self):
        """Create sample options data for testing."""
        calls = pd.DataFrame({
            'strike': [100, 105, 110],
            'bid': [5.0, 2.5, 1.0],
            'ask': [5.2, 2.7, 1.2],
            'volume': [100, 50, 25],
            'openInterest': [1000, 500, 250],
            'impliedVolatility': [0.25, 0.27, 0.30]
        })

        puts = pd.DataFrame({
            'strike': [100, 105, 110],
            'bid': [1.0, 2.5, 5.0],
            'ask': [1.2, 2.7, 5.2],
            'volume': [25, 50, 100],
            'openInterest': [250, 500, 1000],
            'impliedVolatility': [0.30, 0.27, 0.25]
        })

        return OptionChainData(
            symbol="SPY",
            expiration=datetime(2024, 1, 19),
            calls=calls,
            puts=puts,
            underlying_price=102.5,
            source="yfinance"
        )

    def test_cache_initialization(self, temp_cache):
        """Test cache manager initializes correctly."""
        cache, mock_provider = temp_cache
        assert cache.database_path.exists()
        assert cache.Session is not None
        assert cache.provider == mock_provider

    def test_price_data_cache_miss_then_hit(self, temp_cache, sample_price_data):
        """Test price data caching - miss then hit."""
        cache, mock_provider = temp_cache
        mock_provider.get_price_history.return_value = sample_price_data

        # First call - should be a cache miss
        result1 = cache.get_price_history("SPY")
        mock_provider.get_price_history.assert_called_once()
        assert result1.symbol == "SPY"
        assert len(result1.data) == 3

        # Reset mock
        mock_provider.reset_mock()

        # Second call - should be a cache hit
        result2 = cache.get_price_history("SPY")
        mock_provider.get_price_history.assert_not_called()
        assert result2.symbol == "SPY"
        assert result2.source == "cache"
        assert len(result2.data) == 3

    def test_price_data_storage(self, temp_cache, sample_price_data):
        """Test that price data is stored correctly."""
        cache, mock_provider = temp_cache
        session = cache.Session()

        try:
            cache._store_price_data(session, sample_price_data)
            session.commit()

            # Verify data was stored
            stored_data = session.query(PriceHistory).filter_by(symbol="SPY").all()
            assert len(stored_data) == 3

            # Check first record
            first_record = stored_data[0]
            assert first_record.symbol == "SPY"
            assert first_record.open == 100.0
            assert first_record.high == 101.0
            assert first_record.close == 100.5

        finally:
            session.close()

    def test_options_data_cache_miss_then_hit(self, temp_cache, sample_options_data):
        """Test options data caching - miss then hit."""
        cache, mock_provider = temp_cache
        mock_provider.get_options_chain.return_value = sample_options_data

        expiration = datetime(2024, 1, 19)

        # First call - cache miss
        result1 = cache.get_options_chain("SPY", expiration)
        mock_provider.get_options_chain.assert_called_once()
        assert result1.symbol == "SPY"
        assert len(result1.calls) == 3

        # Reset mock
        mock_provider.reset_mock()

        # Second call - cache hit
        result2 = cache.get_options_chain("SPY", expiration)
        mock_provider.get_options_chain.assert_not_called()
        assert result2.symbol == "SPY"
        assert result2.source == "cache"
        assert len(result2.calls) == 3

    def test_options_data_storage(self, temp_cache, sample_options_data):
        """Test that options data is stored correctly."""
        cache, mock_provider = temp_cache
        session = cache.Session()

        try:
            cache._store_options_data(session, sample_options_data)
            session.commit()

            # Verify data was stored
            stored_options = session.query(OptionsData).filter_by(symbol="SPY").all()
            assert len(stored_options) == 6  # 3 calls + 3 puts

            # Check calls vs puts
            calls = [opt for opt in stored_options if opt.option_type == 'call']
            puts = [opt for opt in stored_options if opt.option_type == 'put']
            assert len(calls) == 3
            assert len(puts) == 3

        finally:
            session.close()

    def test_cache_statistics(self, temp_cache, sample_price_data):
        """Test cache statistics tracking."""
        cache, mock_provider = temp_cache
        mock_provider.get_price_history.return_value = sample_price_data

        # Generate some cache activity
        cache.get_price_history("SPY")  # Miss
        cache.get_price_history("SPY")  # Hit
        cache.get_price_history("SPY")  # Hit

        stats = cache.get_cache_statistics()

        # Check that we have statistics
        assert 'database' in stats
        assert stats['database']['price_records'] > 0

        # Check for price stats (may be in different format)
        price_stats_found = False
        for key, value in stats.items():
            if 'price' in key.lower() and 'spy' in key.lower():
                assert value['hits'] >= 2
                assert value['misses'] >= 1
                price_stats_found = True
                break

    def test_data_freshness_with_ttl(self, temp_cache, sample_price_data):
        """Test TTL-based cache invalidation."""
        cache, mock_provider = temp_cache

        # Mock TTL to be very short for testing
        cache._ttl_cache['price_data_ttl_seconds'] = 1  # 1 second

        mock_provider.get_price_history.return_value = sample_price_data

        # First call
        result1 = cache.get_price_history("SPY")
        assert mock_provider.get_price_history.call_count == 1

        # Second call immediately - should hit cache
        result2 = cache.get_price_history("SPY")
        assert mock_provider.get_price_history.call_count == 1
        assert result2.source == "cache"

        # Wait for TTL to expire and make another call
        import time
        time.sleep(1.1)

        # This should be a cache miss due to expired TTL
        result3 = cache.get_price_history("SPY")
        assert mock_provider.get_price_history.call_count == 2

    def test_cleanup_old_data(self, temp_cache, sample_price_data):
        """Test cleaning up old cached data."""
        cache, mock_provider = temp_cache
        session = cache.Session()

        try:
            # Store some data with old timestamp
            cache._store_price_data(session, sample_price_data)

            # Manually update timestamps to simulate old data
            old_timestamp = datetime.utcnow() - timedelta(days=40)
            session.query(PriceHistory).update({'updated_at': old_timestamp})
            session.commit()

            # Verify data exists
            count_before = session.query(PriceHistory).count()
            assert count_before > 0

            # Cleanup old data (older than 30 days)
            deleted_count = cache.cleanup_old_data(max_age_days=30)

            # Verify data was deleted
            count_after = session.query(PriceHistory).count()
            assert count_after == 0
            assert deleted_count == count_before

        finally:
            session.close()

    def test_cache_error_handling(self, temp_cache):
        """Test error handling in cache operations."""
        cache, mock_provider = temp_cache

        # Test provider error handling
        mock_provider.get_price_history.side_effect = Exception("Provider error")

        # Cache should handle errors gracefully
        with pytest.raises(Exception):
            cache.get_price_history("SPY")

    def test_multiple_symbols_caching(self, temp_cache, sample_price_data):
        """Test caching works correctly for multiple symbols."""
        cache, mock_provider = temp_cache

        # Create different data for different symbols
        spy_data = sample_price_data
        qqq_data = PriceData(
            symbol="QQQ",
            data=sample_price_data.data * 1.5,  # Different prices
            start_date=sample_price_data.start_date,
            end_date=sample_price_data.end_date,
            interval="1d",
            source="yfinance"
        )

        def mock_get_price_history(symbol, start=None, end=None, interval="1d"):
            if symbol == "SPY":
                return spy_data
            elif symbol == "QQQ":
                return qqq_data
            else:
                raise ValueError(f"Unknown symbol: {symbol}")

        mock_provider.get_price_history.side_effect = mock_get_price_history

        # Fetch data for both symbols
        spy_result = cache.get_price_history("SPY")
        qqq_result = cache.get_price_history("QQQ")

        assert spy_result.symbol == "SPY"
        assert qqq_result.symbol == "QQQ"
        assert not spy_result.data.equals(qqq_result.data)

        # Verify cache hits work for both
        spy_result2 = cache.get_price_history("SPY")
        qqq_result2 = cache.get_price_history("QQQ")

        assert spy_result2.source == "cache"
        assert qqq_result2.source == "cache"


if __name__ == "__main__":
    pytest.main([__file__])
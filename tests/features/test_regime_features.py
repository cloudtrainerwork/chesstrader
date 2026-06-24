"""
Tests for regime feature engineering components.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from src.features.regime_features import (
    PriceStructureFeatures,
    TrendIndicators,
    MomentumIndicators,
    VolatilityFeatures,
    RegimeStateVector
)


class TestPriceStructureFeatures:
    """Test price structure features calculation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        np.random.seed(42)  # For reproducible tests

        # Create realistic price data with trend
        base_price = 100
        trend = np.linspace(0, 20, 300)
        noise = np.random.normal(0, 2, 300)
        close_prices = base_price + trend + noise

        # Create OHLC from close
        opens = close_prices * (1 + np.random.normal(0, 0.01, 300))
        highs = np.maximum(opens, close_prices) * (1 + np.abs(np.random.normal(0, 0.01, 300)))
        lows = np.minimum(opens, close_prices) * (1 - np.abs(np.random.normal(0, 0.01, 300)))
        volumes = np.random.randint(1000000, 10000000, 300)

        return pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': close_prices,
            'Volume': volumes
        }, index=dates)

    @pytest.fixture
    def price_features(self):
        """Create PriceStructureFeatures instance with mocked data access."""
        pf = PriceStructureFeatures()
        return pf

    def test_price_structure_features_shape(self, price_features, sample_data):
        """Test that price structure features return correct shape."""
        with patch.object(price_features, 'get_data', return_value=sample_data):
            features = price_features.calculate('SPY')

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == 6, f"Expected 6 features, got {features.shape[1]}"
        assert len(features) == len(sample_data)

    def test_price_structure_features_columns(self, price_features, sample_data):
        """Test that price structure features have correct columns."""
        expected_columns = [
            'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200',
            'distance_from_52w_high', 'distance_from_52w_low', 'gap_percentage'
        ]

        with patch.object(price_features, 'get_data', return_value=sample_data):
            features = price_features.calculate('SPY')

        assert list(features.columns) == expected_columns

    def test_price_structure_no_nans(self, price_features, sample_data):
        """Test that price structure features have no NaN values."""
        with patch.object(price_features, 'get_data', return_value=sample_data):
            features = price_features.calculate('SPY')

        assert not features.isnull().any().any(), "Features contain NaN values"

    def test_price_structure_normalized_range(self, price_features, sample_data):
        """Test that price structure features are in reasonable range."""
        with patch.object(price_features, 'get_data', return_value=sample_data):
            features = price_features.calculate('SPY')

        # Check that values are roughly in [-3, 3] range (allowing some outliers)
        for col in features.columns:
            assert features[col].min() > -5, f"Column {col} has extreme negative value"
            assert features[col].max() < 5, f"Column {col} has extreme positive value"


class TestTrendIndicators:
    """Test trend indicators calculation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        np.random.seed(42)

        base_price = 100
        trend = np.linspace(0, 20, 300)
        noise = np.random.normal(0, 1, 300)
        close_prices = base_price + trend + noise

        opens = close_prices * (1 + np.random.normal(0, 0.005, 300))
        highs = np.maximum(opens, close_prices) * (1 + np.abs(np.random.normal(0, 0.01, 300)))
        lows = np.minimum(opens, close_prices) * (1 - np.abs(np.random.normal(0, 0.01, 300)))
        volumes = np.random.randint(1000000, 10000000, 300)

        return pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': close_prices,
            'Volume': volumes
        }, index=dates)

    @pytest.fixture
    def trend_features(self):
        """Create TrendIndicators instance with mocked data access."""
        tf = TrendIndicators()
        return tf

    def test_trend_indicators_shape(self, trend_features, sample_data):
        """Test that trend indicators return correct shape."""
        with patch.object(trend_features, 'get_data', return_value=sample_data):
            features = trend_features.calculate('SPY')

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == 9, f"Expected 9 features, got {features.shape[1]}"

    def test_trend_indicators_adx_calculation(self, trend_features, sample_data):
        """Test ADX calculation specifically."""
        with patch.object(trend_features, 'get_data', return_value=sample_data):
            adx, di_plus, di_minus = trend_features.calculate_adx(
                sample_data['High'], sample_data['Low'], sample_data['Close']
            )

        assert not adx.isnull().all(), "ADX should not be all NaN"
        assert not di_plus.isnull().all(), "+DI should not be all NaN"
        assert not di_minus.isnull().all(), "-DI should not be all NaN"

    def test_trend_indicators_no_nans(self, trend_features, sample_data):
        """Test that trend indicators have no NaN values after processing."""
        with patch.object(trend_features, 'get_data', return_value=sample_data):
            features = trend_features.calculate('SPY')

        assert not features.isnull().any().any(), "Features contain NaN values"


class TestMomentumIndicators:
    """Test momentum indicators calculation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        np.random.seed(42)

        base_price = 100
        trend = np.sin(np.linspace(0, 4*np.pi, 300)) * 10  # Oscillating prices
        noise = np.random.normal(0, 1, 300)
        close_prices = base_price + trend + noise

        opens = close_prices * (1 + np.random.normal(0, 0.005, 300))
        highs = np.maximum(opens, close_prices) * (1 + np.abs(np.random.normal(0, 0.01, 300)))
        lows = np.minimum(opens, close_prices) * (1 - np.abs(np.random.normal(0, 0.01, 300)))
        volumes = np.random.randint(1000000, 10000000, 300)

        return pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': close_prices,
            'Volume': volumes
        }, index=dates)

    @pytest.fixture
    def momentum_features(self):
        """Create MomentumIndicators instance with mocked data access."""
        mf = MomentumIndicators()
        return mf

    def test_momentum_indicators_shape(self, momentum_features, sample_data):
        """Test that momentum indicators return correct shape."""
        with patch.object(momentum_features, 'get_data', return_value=sample_data):
            features = momentum_features.calculate('SPY')

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == 6, f"Expected 6 features, got {features.shape[1]}"

    def test_rsi_calculation(self, momentum_features, sample_data):
        """Test RSI calculation specifically."""
        rsi = momentum_features.calculate_rsi(sample_data['Close'])

        assert not rsi.isnull().all(), "RSI should not be all NaN"
        # RSI normalized values should be roughly in [-1, 1]
        normalized_rsi = (rsi / 50) - 1
        assert normalized_rsi.max() <= 1.1, "Normalized RSI too high"
        assert normalized_rsi.min() >= -1.1, "Normalized RSI too low"

    def test_stochastic_calculation(self, momentum_features, sample_data):
        """Test Stochastic calculation."""
        k, d = momentum_features.calculate_stochastic(
            sample_data['High'], sample_data['Low'], sample_data['Close']
        )

        assert not k.isnull().all(), "Stochastic %K should not be all NaN"
        assert not d.isnull().all(), "Stochastic %D should not be all NaN"


class TestRegimeStateVector:
    """Test complete regime state vector assembly."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        np.random.seed(42)

        base_price = 100
        trend = np.linspace(0, 20, 300)
        noise = np.random.normal(0, 2, 300)
        close_prices = base_price + trend + noise

        opens = close_prices * (1 + np.random.normal(0, 0.01, 300))
        highs = np.maximum(opens, close_prices) * (1 + np.abs(np.random.normal(0, 0.01, 300)))
        lows = np.minimum(opens, close_prices) * (1 - np.abs(np.random.normal(0, 0.01, 300)))
        volumes = np.random.randint(1000000, 10000000, 300)

        return pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': close_prices,
            'Volume': volumes
        }, index=dates)

    def test_regime_state_vector_partial(self, sample_data):
        """Test regime state vector with current implementation (21 dimensions)."""
        rsv = RegimeStateVector()

        # Mock the get_data method for all sub-components
        with patch.object(rsv.price_features, 'get_data', return_value=sample_data), \
             patch.object(rsv.trend_features, 'get_data', return_value=sample_data), \
             patch.object(rsv.momentum_features, 'get_data', return_value=sample_data):

            state = rsv.calculate('SPY')

        assert isinstance(state, pd.Series)
        # Currently we have 6 + 9 + 6 = 21 dimensions (Tasks 1 & 2 complete)
        assert len(state) == 21, f"Expected 21 dimensions, got {len(state)}"

    def test_regime_state_vector_no_nans(self, sample_data):
        """Test that regime state vector has no NaN values."""
        rsv = RegimeStateVector()

        with patch.object(rsv.price_features, 'get_data', return_value=sample_data), \
             patch.object(rsv.trend_features, 'get_data', return_value=sample_data), \
             patch.object(rsv.momentum_features, 'get_data', return_value=sample_data):

            state = rsv.calculate('SPY')

        assert not state.isnull().any(), "State vector contains NaN values"


class TestVolatilityFeatures:
    """Verify live-signal volatility features replaced the hardcoded stubs."""

    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range('2022-01-01', periods=300, freq='D')
        np.random.seed(7)
        close = 100 + np.cumsum(np.random.normal(0, 1, 300))
        high = close * (1 + np.abs(np.random.normal(0, 0.01, 300)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, 300)))
        return pd.DataFrame({
            'Open': close, 'High': high, 'Low': low,
            'Close': close, 'Volume': np.ones(300) * 1e6,
        }, index=dates)

    def _fake_vix(self, dates):
        """VIX that varies so we can assert it's not the constant 20.0."""
        vix = pd.Series(np.linspace(15, 35, len(dates)), index=dates)
        return pd.DataFrame({'vix': vix.values, 'vix_pct': (vix / 35).values}, index=dates)

    def test_vix_level_varies(self, sample_data):
        """vix_level must not be a constant — it now comes from live ^VIX data."""
        vf = VolatilityFeatures()
        with patch.object(vf, '_get_vix_aligned', side_effect=self._fake_vix):
            feats = vf.calculate('SPY', data=sample_data)
        assert feats['vix_level'].nunique() > 1, "vix_level is still a constant stub"

    def test_implied_volatility_varies(self, sample_data):
        """implied_volatility must not be the hardcoded 0.2 constant."""
        vf = VolatilityFeatures()
        with patch.object(vf, '_get_vix_aligned', side_effect=self._fake_vix):
            feats = vf.calculate('SPY', data=sample_data)
        assert feats['implied_volatility'].nunique() > 1, "implied_volatility is still a constant stub"

    def test_iv_rank_varies(self, sample_data):
        """iv_rank must not be the hardcoded 0.5 constant."""
        vf = VolatilityFeatures()
        with patch.object(vf, '_get_vix_aligned', side_effect=self._fake_vix):
            feats = vf.calculate('SPY', data=sample_data)
        assert feats['iv_rank'].nunique() > 1, "iv_rank is still a constant stub"

    def test_no_nans(self, sample_data):
        """Feature DataFrame must have no NaN after calculation."""
        vf = VolatilityFeatures()
        with patch.object(vf, '_get_vix_aligned', side_effect=self._fake_vix):
            feats = vf.calculate('SPY', data=sample_data)
        assert not feats.isnull().any().any()
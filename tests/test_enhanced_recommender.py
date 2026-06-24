"""
Tests for EnhancedStrategyRecommender ML-pipeline wiring (issue #5).

Verifies that:
  - _detect_regime falls back to heuristic when no checkpoint exists
  - _heuristic_regime maps SMA sentiment to the correct RegimeType ints
  - get_actionable_recommendations drives strategy selection via StrategyFactory
    rather than the old hardcoded strategy-type list
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.api.enhanced_strategy_recommender import EnhancedStrategyRecommender
from src.data.regime_labeler import RegimeType
from src.strategies.base import StrategyType


def _make_hist(n=60, trend='flat'):
    """Synthetic OHLCV DataFrame."""
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    if trend == 'bullish':
        close = np.linspace(100, 120, n)
    elif trend == 'bearish':
        close = np.linspace(120, 100, n)
    else:
        close = np.full(n, 110.0)
    return pd.DataFrame({
        'Open': close, 'High': close * 1.01, 'Low': close * 0.99,
        'Close': close, 'Volume': np.ones(n) * 1e6,
    }, index=dates)


class TestHeuristicRegime:
    def test_bullish_sentiment_maps_to_bull_trending(self):
        rec = EnhancedStrategyRecommender()
        with patch.object(rec, '_analyze_market_sentiment', return_value='bullish'):
            regime = rec._heuristic_regime(_make_hist())
        assert regime == int(RegimeType.BULL_TRENDING)

    def test_bearish_sentiment_maps_to_bear_trending(self):
        rec = EnhancedStrategyRecommender()
        with patch.object(rec, '_analyze_market_sentiment', return_value='bearish'):
            regime = rec._heuristic_regime(_make_hist())
        assert regime == int(RegimeType.BEAR_TRENDING)

    def test_neutral_sentiment_maps_to_sideways(self):
        rec = EnhancedStrategyRecommender()
        with patch.object(rec, '_analyze_market_sentiment', return_value='neutral'):
            regime = rec._heuristic_regime(_make_hist())
        assert regime == int(RegimeType.SIDEWAYS_RANGING)


class TestDetectRegime:
    def test_no_checkpoint_uses_heuristic(self):
        rec = EnhancedStrategyRecommender()
        hist = _make_hist()
        with patch.object(rec, '_find_checkpoint', return_value=None), \
             patch.object(rec, '_heuristic_regime', return_value=int(RegimeType.BULL_TRENDING)) as mock_h:
            regime = rec._detect_regime('SPY', hist)
        mock_h.assert_called_once_with(hist)
        assert regime == int(RegimeType.BULL_TRENDING)

    def test_ml_failure_falls_back_to_heuristic(self):
        rec = EnhancedStrategyRecommender()
        hist = _make_hist()
        with patch.object(rec, '_find_checkpoint', return_value='/fake/checkpoint.pt'), \
             patch('torch.load', side_effect=RuntimeError("corrupt")), \
             patch.object(rec, '_heuristic_regime', return_value=int(RegimeType.BEAR_TRENDING)) as mock_h:
            regime = rec._detect_regime('SPY', hist)
        mock_h.assert_called_once()
        assert regime == int(RegimeType.BEAR_TRENDING)


class TestGetActionableRecommendations:
    """Verify factory drives strategy selection instead of hardcoded strings."""

    def _mock_yf(self, hist, expirations, chain):
        ticker = MagicMock()
        ticker.history.return_value = hist
        ticker.options = expirations
        ticker.option_chain.return_value = chain
        return ticker

    def _fake_chain(self, price=100.0):
        strikes = np.arange(price * 0.80, price * 1.20, price * 0.05)
        calls = pd.DataFrame({
            'strike': strikes, 'bid': 2.0, 'ask': 2.5, 'lastPrice': 2.25,
            'impliedVolatility': 0.25,
        })
        puts = calls.copy()
        return MagicMock(calls=calls, puts=puts)

    def test_factory_is_called_with_detected_regime(self):
        rec = EnhancedStrategyRecommender()
        hist = _make_hist(60, 'bullish')
        chain = self._fake_chain()
        mock_ticker = self._mock_yf(hist, ['2026-08-15'], chain)

        with patch('yfinance.Ticker', return_value=mock_ticker), \
             patch.object(rec, '_detect_regime', return_value=int(RegimeType.BULL_TRENDING)) as mock_detect, \
             patch('src.strategies.factory.StrategyFactory') as MockFactory:

            factory_inst = MagicMock()
            MockFactory.return_value = factory_inst
            fake_rec = MagicMock()
            fake_rec.strategy_type = StrategyType.BULL_CALL_SPREAD
            factory_inst.get_recommended_strategies.return_value = [fake_rec]

            # Also patch the module-level import that EnhancedStrategyRecommender uses
            with patch('src.api.enhanced_strategy_recommender.StrategyFactory', MockFactory):
                recs = rec.get_actionable_recommendations('SPY')

        mock_detect.assert_called_once()
        factory_inst.get_recommended_strategies.assert_called_once_with(
            int(RegimeType.BULL_TRENDING), max_recommendations=5
        )

    def test_returns_list_on_success(self):
        rec = EnhancedStrategyRecommender()
        hist = _make_hist(60, 'flat')
        chain = self._fake_chain()
        mock_ticker = self._mock_yf(hist, ['2026-08-15'], chain)

        with patch('yfinance.Ticker', return_value=mock_ticker), \
             patch.object(rec, '_detect_regime', return_value=int(RegimeType.SIDEWAYS_RANGING)), \
             patch('src.api.enhanced_strategy_recommender.StrategyFactory') as MockFactory:

            factory_inst = MagicMock()
            MockFactory.return_value = factory_inst
            fake_rec = MagicMock()
            fake_rec.strategy_type = StrategyType.IRON_CONDOR
            factory_inst.get_recommended_strategies.return_value = [fake_rec]

            recs = rec.get_actionable_recommendations('SPY')

        assert isinstance(recs, list)

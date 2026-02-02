"""Tests for recommendation engine."""

import pytest
import numpy as np
import torch
from typing import Dict, List
from datetime import datetime

from src.models.recommendation_engine import (
    RecommendationEngine,
    StrategyRecommendation
)
from src.models.integrated_selector import IntegratedStrategySelector
from src.models.strategy_selector import StrategySelector
from src.models.spatial_net import SpatialNet
from src.models.scoring_engine import ScoringEngine
from src.features.regime_detector import RegimeDetector, RegimeType
from src.strategies.base import StrategyType
from src.api.strategy_recommender import StrategyRecommender


class TestRecommendationEngine:
    """Test recommendation engine functionality."""

    @pytest.fixture
    def spatial_net(self):
        """Create spatial network."""
        return SpatialNet(
            board_height=7,
            board_width=6,
            piece_channels=16,
            spatial_hidden_dim=256,
            output_dim=512
        )

    @pytest.fixture
    def strategy_selector(self, spatial_net):
        """Create strategy selector."""
        return StrategySelector(
            spatial_net=spatial_net,
            freeze_spatial_net=False
        )

    @pytest.fixture
    def regime_detector(self):
        """Create regime detector."""
        return RegimeDetector(
            n_regimes=5,
            lookback_period=20,
            hidden_dim=128
        )

    @pytest.fixture
    def integrated_selector(self, strategy_selector, regime_detector):
        """Create integrated selector."""
        return IntegratedStrategySelector(
            strategy_selector=strategy_selector,
            regime_detector=regime_detector
        )

    @pytest.fixture
    def scoring_engine(self):
        """Create scoring engine."""
        return ScoringEngine(
            kelly_fraction=0.25,
            min_confidence=0.3
        )

    @pytest.fixture
    def recommendation_engine(self, integrated_selector, scoring_engine):
        """Create recommendation engine."""
        return RecommendationEngine(
            integrated_selector=integrated_selector,
            scoring_engine=scoring_engine,
            confidence_threshold=0.4,
            max_recommendations=3
        )

    @pytest.fixture
    def sample_market_data(self):
        """Sample market data."""
        prices = np.array([100, 101, 99, 102, 103, 101, 100, 102, 104, 103])
        returns = np.diff(prices) / prices[:-1]

        return {
            'symbol': 'TEST',
            'current_price': 103.0,
            'price_history': prices,
            'returns': returns,
            'volume': 1000000,
            'implied_volatility': 0.25,
            'historical_volatility': np.std(returns) * np.sqrt(252)
        }

    @pytest.fixture
    def sample_regime_features(self):
        """Sample regime features."""
        return np.random.randn(20)  # 20 features for regime detection

    def test_get_top_recommendations(self, recommendation_engine,
                                    sample_market_data, sample_regime_features):
        """Test getting top recommendations."""
        recommendations = recommendation_engine.get_top_recommendations(
            sample_market_data,
            sample_regime_features
        )

        # Check we get recommendations
        assert len(recommendations) <= recommendation_engine.max_recommendations
        assert len(recommendations) > 0

        # Check all are StrategyRecommendation instances
        assert all(isinstance(r, StrategyRecommendation) for r in recommendations)

        # Check recommendations have required fields
        for rec in recommendations:
            assert rec.strategy_type in StrategyType
            assert 0 <= rec.score <= 100
            assert 0 <= rec.confidence <= 1
            assert 0 <= rec.position_size <= 0.2  # Max position size
            assert rec.regime in RegimeType
            assert rec.explanation

    def test_confidence_filtering(self, integrated_selector, scoring_engine):
        """Test confidence threshold filtering."""
        # Create engine with high confidence threshold
        high_threshold_engine = RecommendationEngine(
            integrated_selector=integrated_selector,
            scoring_engine=scoring_engine,
            confidence_threshold=0.8,  # Very high threshold
            max_recommendations=3
        )

        market_data = {'current_price': 100, 'volume': 1000000}
        regime_features = np.random.randn(20)

        recommendations = high_threshold_engine.get_top_recommendations(
            market_data,
            regime_features
        )

        # With high threshold, we might get fewer or no recommendations
        # (depending on model outputs)
        assert len(recommendations) <= 3

    def test_regime_validation(self, recommendation_engine):
        """Test strategy validation against market regime."""
        # Test trending up regime
        market_data = {'current_price': 100}
        regime_features = np.array([0.5] * 20)  # Features suggesting uptrend

        recommendations = recommendation_engine.get_top_recommendations(
            market_data,
            regime_features
        )

        # Should get some recommendations
        assert len(recommendations) > 0

    def test_batch_recommend(self, recommendation_engine):
        """Test batch recommendations."""
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        market_data_batch = [
            {'current_price': 150, 'volume': 10000000},
            {'current_price': 2800, 'volume': 5000000},
            {'current_price': 300, 'volume': 8000000}
        ]
        regime_features_batch = [
            np.random.randn(20),
            np.random.randn(20),
            np.random.randn(20)
        ]

        batch_results = recommendation_engine.batch_recommend(
            symbols,
            market_data_batch,
            regime_features_batch
        )

        # Check we get results for all symbols
        assert len(batch_results) == len(symbols)
        assert all(symbol in batch_results for symbol in symbols)

        # Check each symbol has recommendations
        for symbol, recommendations in batch_results.items():
            assert isinstance(recommendations, list)
            assert len(recommendations) <= recommendation_engine.max_recommendations

    def test_recommendation_explanation(self, recommendation_engine,
                                       sample_market_data, sample_regime_features):
        """Test recommendation explanations."""
        recommendations = recommendation_engine.get_top_recommendations(
            sample_market_data,
            sample_regime_features
        )

        for rec in recommendations:
            # Check explanation contains key information
            assert rec.strategy_type.value in rec.explanation
            assert "Score:" in rec.explanation
            assert "Market regime:" in rec.explanation
            assert "Expected return:" in rec.explanation
            assert "position size:" in rec.explanation

    def test_recommendation_metadata(self, recommendation_engine,
                                    sample_market_data, sample_regime_features):
        """Test recommendation metadata."""
        recommendations = recommendation_engine.get_top_recommendations(
            sample_market_data,
            sample_regime_features
        )

        for rec in recommendations:
            # Check metadata fields
            assert 'regime_confidence' in rec.metadata
            assert 'neural_network_probability' in rec.metadata
            assert 'rank' in rec.metadata
            assert 'market_data_snapshot' in rec.metadata

            # Check timestamp
            assert isinstance(rec.timestamp, datetime)

    def test_history_tracking(self, recommendation_engine,
                             sample_market_data, sample_regime_features):
        """Test recommendation history tracking."""
        # Get initial recommendations
        recommendations1 = recommendation_engine.get_top_recommendations(
            sample_market_data,
            sample_regime_features
        )

        # Get more recommendations
        recommendations2 = recommendation_engine.get_top_recommendations(
            sample_market_data,
            sample_regime_features
        )

        # Check history is tracked
        history = recommendation_engine.get_recommendation_history()
        assert len(history) == len(recommendations1) + len(recommendations2)

        # Clear history
        recommendation_engine.clear_history()
        assert len(recommendation_engine.get_recommendation_history()) == 0


class TestStrategyRecommenderAPI:
    """Test high-level API interface."""

    @pytest.fixture
    def recommender_api(self):
        """Create recommender API instance."""
        return StrategyRecommender(
            confidence_threshold=0.4,
            max_recommendations=3
        )

    def test_recommend_simple(self, recommender_api):
        """Test simple recommendation call."""
        price_history = [100, 101, 99, 102, 103, 101, 100, 102, 104, 103,
                        105, 104, 106, 105, 107, 106, 108, 107, 109, 108]

        recommendations = recommender_api.recommend(
            symbol='TEST',
            current_price=108.0,
            price_history=price_history,
            implied_volatility=0.25
        )

        # Check format
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3

        for rec in recommendations:
            assert 'strategy' in rec
            assert 'score' in rec
            assert 'confidence' in rec
            assert 'position_size' in rec
            assert 'expected_return' in rec
            assert 'max_risk' in rec
            assert 'regime' in rec
            assert 'explanation' in rec

    def test_batch_recommend_api(self, recommender_api):
        """Test batch recommendations through API."""
        symbols = ['AAPL', 'GOOGL']
        prices = [150.0, 2800.0]
        price_histories = [
            list(range(140, 151)),
            list(range(2750, 2801, 5))
        ]

        batch_results = recommender_api.batch_recommend(
            symbols=symbols,
            prices=prices,
            price_histories=price_histories
        )

        assert len(batch_results) == 2
        assert 'AAPL' in batch_results
        assert 'GOOGL' in batch_results

    def test_strategy_details(self, recommender_api):
        """Test getting strategy details."""
        details = recommender_api.get_strategy_details('Iron Condor')

        assert 'name' in details
        assert 'description' in details
        assert 'risk_profile' in details
        assert 'optimal_conditions' in details
        assert 'typical_returns' in details

        # Test invalid strategy
        invalid_details = recommender_api.get_strategy_details('Invalid Strategy')
        assert 'error' in invalid_details

    def test_market_data_preparation(self, recommender_api):
        """Test market data preparation."""
        market_data = recommender_api._prepare_market_data(
            symbol='TEST',
            current_price=100.0,
            price_history=[95, 96, 97, 98, 99, 100],
            volume_history=[1000000, 1100000, 950000, 1050000, 1000000, 1200000],
            implied_volatility=0.25,
            option_chain=None
        )

        assert market_data['symbol'] == 'TEST'
        assert market_data['current_price'] == 100.0
        assert 'returns' in market_data
        assert 'historical_volatility' in market_data
        assert market_data['implied_volatility'] == 0.25
        assert market_data['has_options'] == False

    def test_regime_feature_extraction(self, recommender_api):
        """Test regime feature extraction."""
        price_history = list(range(100, 120))
        features = recommender_api._extract_regime_features(
            price_history,
            volume_history=None
        )

        # Check feature dimensions
        assert features.shape == (20,)
        assert not np.isnan(features).any()

    def test_update_historical_performance(self, recommender_api):
        """Test updating historical performance."""
        performance_data = {
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.08,
            'win_rate': 0.65
        }

        # Should not raise error
        recommender_api.update_historical_performance(
            'Iron Condor',
            performance_data
        )

        # Test invalid strategy
        recommender_api.update_historical_performance(
            'Invalid Strategy',
            performance_data
        )  # Should log error but not raise
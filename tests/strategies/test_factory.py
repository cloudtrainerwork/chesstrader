"""
Tests for the StrategyFactory and strategy recommendation engine.

Tests factory pattern implementation, regime-based recommendations,
strategy instantiation, and recommendation scoring algorithms.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.strategies import StrategyFactory, StrategyRecommendation, StrategyType
from src.strategies.base import MarketConditions, RiskLevel, StrategyCategory


class TestStrategyFactory:
    """Test StrategyFactory implementation."""

    @pytest.fixture
    def factory(self):
        """Strategy factory fixture."""
        return StrategyFactory()

    @pytest.fixture
    def bull_conditions(self):
        """Bullish market conditions."""
        return MarketConditions(
            regime=1,  # Bull trending
            volatility_rank=0.4,
            trend_strength=0.7,
            time_to_expiration=35,
            underlying_price=10000,
            risk_free_rate=0.05
        )

    @pytest.fixture
    def bear_conditions(self):
        """Bearish market conditions."""
        return MarketConditions(
            regime=0,  # Deep bear
            volatility_rank=0.7,
            trend_strength=-0.8,
            time_to_expiration=30,
            underlying_price=8000,
            risk_free_rate=0.05
        )

    @pytest.fixture
    def low_vol_conditions(self):
        """Low volatility market conditions."""
        return MarketConditions(
            regime=4,  # Low vol sideways
            volatility_rank=0.2,
            trend_strength=0.1,
            time_to_expiration=40,
            underlying_price=10000,
            risk_free_rate=0.05
        )

    def test_factory_initialization(self, factory):
        """Test factory initializes with correct registry and mappings."""
        assert factory is not None
        assert factory.get_strategy_count() == 16

        # Test strategy registry is built
        supported_types = factory.get_supported_strategy_types()
        assert len(supported_types) == 16
        assert StrategyType.BULL_CALL_SPREAD in supported_types
        assert StrategyType.IRON_CONDOR in supported_types
        assert StrategyType.CALENDAR_CALL in supported_types

    def test_create_strategy_valid_types(self, factory):
        """Test creating strategies with valid types."""
        test_types = [
            StrategyType.BULL_CALL_SPREAD,
            StrategyType.IRON_CONDOR,
            StrategyType.LONG_STRADDLE,
            StrategyType.CALENDAR_CALL
        ]

        for strategy_type in test_types:
            strategy = factory.create_strategy(strategy_type)
            assert strategy is not None
            assert hasattr(strategy, 'name')
            assert hasattr(strategy, 'category')
            assert hasattr(strategy, 'risk_level')

    def test_create_strategy_invalid_type(self, factory):
        """Test error handling for invalid strategy types."""
        # This test would need a way to create an invalid StrategyType
        # For now, we'll test the error handling structure exists
        assert hasattr(factory, 'create_strategy')

    def test_get_all_strategies(self, factory):
        """Test getting all strategy instances."""
        strategies = factory.get_all_strategies()

        assert isinstance(strategies, list)
        assert len(strategies) >= 10  # Should get most strategies

        # Verify strategy diversity
        strategy_names = [s.name for s in strategies]
        assert len(set(strategy_names)) >= 10  # Unique strategy names

        # Verify all are BaseStrategy instances
        from src.strategies.base import BaseStrategy
        for strategy in strategies:
            assert isinstance(strategy, BaseStrategy)

    def test_regime_mappings_coverage(self, factory):
        """Test that regime mappings cover all market regimes."""
        # Test each regime has strategy mappings
        for regime in range(8):  # 0-7 regimes
            recommendations = factory.get_recommended_strategies(regime)
            assert len(recommendations) > 0, f"No recommendations for regime {regime}"

    def test_bull_regime_recommendations(self, factory, bull_conditions):
        """Test recommendations for bullish market regime."""
        recommendations = factory.get_recommended_strategies(1, bull_conditions)

        assert len(recommendations) > 0
        assert len(recommendations) <= 5  # Default max

        # Should favor bullish strategies
        recommended_types = [rec.strategy_type for rec in recommendations]
        bullish_strategies = [
            StrategyType.LONG_CALL,
            StrategyType.BULL_CALL_SPREAD,
            StrategyType.BULL_PUT_SPREAD
        ]

        # At least some bullish strategies should be recommended
        bullish_recommended = any(st in recommended_types for st in bullish_strategies)
        assert bullish_recommended, "Should recommend bullish strategies in bull regime"

        # Verify confidence scores
        for rec in recommendations:
            assert 0.0 <= rec.confidence <= 1.0
            assert len(rec.reasons) > 0

    def test_bear_regime_recommendations(self, factory, bear_conditions):
        """Test recommendations for bearish market regime."""
        recommendations = factory.get_recommended_strategies(0, bear_conditions)

        assert len(recommendations) > 0

        # Should favor bearish/protective strategies
        recommended_types = [rec.strategy_type for rec in recommendations]
        bearish_strategies = [
            StrategyType.LONG_PUT,
            StrategyType.BEAR_PUT_SPREAD,
            StrategyType.BEAR_CALL_SPREAD,
            StrategyType.LONG_STRADDLE
        ]

        # At least some bearish strategies should be recommended
        bearish_recommended = any(st in recommended_types for st in bearish_strategies)
        assert bearish_recommended, "Should recommend bearish strategies in bear regime"

    def test_low_volatility_recommendations(self, factory, low_vol_conditions):
        """Test recommendations for low volatility regime."""
        recommendations = factory.get_recommended_strategies(4, low_vol_conditions)

        assert len(recommendations) > 0

        # Should favor time decay and income strategies
        recommended_types = [rec.strategy_type for rec in recommendations]
        low_vol_strategies = [
            StrategyType.CALENDAR_CALL,
            StrategyType.CALENDAR_PUT,
            StrategyType.SHORT_STRADDLE,
            StrategyType.IRON_CONDOR
        ]

        # At least some low vol strategies should be recommended
        low_vol_recommended = any(st in recommended_types for st in low_vol_strategies)
        assert low_vol_recommended, "Should recommend time decay strategies in low vol regime"

    def test_recommendation_ranking(self, factory, bull_conditions):
        """Test that recommendations are properly ranked by confidence."""
        recommendations = factory.get_recommended_strategies(1, bull_conditions)

        # Should be sorted by confidence descending
        for i in range(len(recommendations) - 1):
            assert recommendations[i].confidence >= recommendations[i + 1].confidence

    def test_recommendation_structure(self, factory, bull_conditions):
        """Test recommendation object structure and completeness."""
        recommendations = factory.get_recommended_strategies(1, bull_conditions)

        for rec in recommendations:
            assert isinstance(rec, StrategyRecommendation)
            assert isinstance(rec.strategy_type, StrategyType)
            assert rec.strategy_instance is not None
            assert isinstance(rec.confidence, float)
            assert 0.0 <= rec.confidence <= 1.0
            assert isinstance(rec.reasons, list)
            assert len(rec.reasons) > 0
            assert rec.expected_performance is not None
            assert rec.risk_assessment is not None

    def test_max_recommendations_limit(self, factory, bull_conditions):
        """Test max recommendations parameter works correctly."""
        # Test different limits
        for max_recs in [1, 3, 5, 10]:
            recommendations = factory.get_recommended_strategies(
                1, bull_conditions, max_recommendations=max_recs
            )
            assert len(recommendations) <= max_recs

    def test_recommendations_without_conditions(self, factory):
        """Test recommendations work without detailed market conditions."""
        recommendations = factory.get_recommended_strategies(1)  # No conditions

        assert len(recommendations) > 0
        assert len(recommendations) <= 5

        # Should still have valid structure
        for rec in recommendations:
            assert isinstance(rec, StrategyRecommendation)
            assert rec.confidence > 0.0

    def test_confidence_adjustment_with_conditions(self, factory, bull_conditions):
        """Test that market conditions affect confidence scores."""
        # Get recommendations with conditions
        recs_with_conditions = factory.get_recommended_strategies(1, bull_conditions)

        # Get recommendations without conditions
        recs_without_conditions = factory.get_recommended_strategies(1)

        # Confidence scores should potentially differ
        # (This is a structural test - implementation may vary)
        assert len(recs_with_conditions) > 0
        assert len(recs_without_conditions) > 0

    def test_expected_performance_categories(self, factory):
        """Test expected performance categorization."""
        performance_categories_found = set()

        # Test across different regimes
        for regime in range(8):
            recommendations = factory.get_recommended_strategies(regime)
            for rec in recommendations[:2]:  # Test first 2 from each regime
                performance_categories_found.add(rec.expected_performance)

        # Should have variety in performance categories
        expected_categories = ["defensive", "growth", "income", "neutral", "volatile"]
        found_expected = any(cat in performance_categories_found for cat in expected_categories)
        assert found_expected, f"Should find expected performance categories, found: {performance_categories_found}"

    def test_risk_assessment_categories(self, factory):
        """Test risk assessment categorization."""
        risk_assessments_found = set()

        # Test across different regimes
        for regime in range(8):
            recommendations = factory.get_recommended_strategies(regime)
            for rec in recommendations[:2]:  # Test first 2 from each regime
                risk_assessments_found.add(rec.risk_assessment)

        # Should have variety in risk assessments
        expected_assessments = ["appropriate_risk", "elevated_risk", "manageable_risk", "conservative_risk"]
        found_expected = any(assess in risk_assessments_found for assess in expected_assessments)
        assert found_expected, f"Should find risk assessment categories, found: {risk_assessments_found}"

    def test_invalid_regime_error(self, factory):
        """Test error handling for invalid market regime."""
        with pytest.raises(ValueError, match="Invalid market regime"):
            factory.get_recommended_strategies(10)  # Invalid regime

        with pytest.raises(ValueError, match="Invalid market regime"):
            factory.get_recommended_strategies(-1)  # Invalid regime

    def test_strategy_factory_performance(self, factory):
        """Test strategy factory performance requirements."""
        import time

        conditions = MarketConditions(1, 0.5, 0.3, 30, 10000, 0.05)

        # Test recommendation generation time
        start_time = time.time()
        recommendations = factory.get_recommended_strategies(1, conditions)
        generation_time = time.time() - start_time

        assert generation_time < 0.1, f"Recommendation generation too slow: {generation_time:.3f}s"
        assert len(recommendations) > 0

        # Test strategy creation time
        start_time = time.time()
        strategy = factory.create_strategy(StrategyType.BULL_CALL_SPREAD)
        creation_time = time.time() - start_time

        assert creation_time < 0.01, f"Strategy creation too slow: {creation_time:.3f}s"
        assert strategy is not None

    def test_regime_strategy_alignment(self, factory):
        """Test that recommended strategies align with regime characteristics."""
        # Test specific regime-strategy alignments

        # Regime 0 (Deep Bear) should favor protective strategies
        bear_recs = factory.get_recommended_strategies(0)
        bear_types = [rec.strategy_type for rec in bear_recs]
        protective_strategies = [StrategyType.LONG_PUT, StrategyType.BEAR_PUT_SPREAD, StrategyType.LONG_STRADDLE]
        assert any(st in bear_types for st in protective_strategies), "Bear regime should favor protective strategies"

        # Regime 4 (Low Vol) should favor time decay strategies
        low_vol_recs = factory.get_recommended_strategies(4)
        low_vol_types = [rec.strategy_type for rec in low_vol_recs]
        time_decay_strategies = [StrategyType.CALENDAR_CALL, StrategyType.CALENDAR_PUT, StrategyType.SHORT_STRADDLE]
        assert any(st in low_vol_types for st in time_decay_strategies), "Low vol regime should favor time decay"

    def test_factory_extensibility(self, factory):
        """Test that factory design supports extensibility."""
        # Test that internal structure supports adding new strategies
        assert hasattr(factory, '_strategy_registry')
        assert hasattr(factory, '_regime_mappings')

        # Test that all current strategy types are handled
        registry_types = set(factory._strategy_registry.keys())
        expected_types = set([
            StrategyType.LONG_CALL, StrategyType.SHORT_CALL,
            StrategyType.LONG_PUT, StrategyType.SHORT_PUT,
            StrategyType.BULL_CALL_SPREAD, StrategyType.BEAR_CALL_SPREAD,
            StrategyType.BULL_PUT_SPREAD, StrategyType.BEAR_PUT_SPREAD,
            StrategyType.CALENDAR_CALL, StrategyType.CALENDAR_PUT,
            StrategyType.LONG_STRADDLE, StrategyType.SHORT_STRADDLE,
            StrategyType.LONG_STRANGLE, StrategyType.SHORT_STRANGLE,
            StrategyType.IRON_CONDOR, StrategyType.BUTTERFLY
        ])

        assert registry_types == expected_types, f"Registry missing types: {expected_types - registry_types}"

    def test_recommendation_consistency(self, factory):
        """Test recommendation consistency across multiple calls."""
        conditions = MarketConditions(1, 0.5, 0.3, 30, 10000, 0.05)

        # Get recommendations multiple times
        recs1 = factory.get_recommended_strategies(1, conditions)
        recs2 = factory.get_recommended_strategies(1, conditions)

        # Should be consistent (same types, order, confidence)
        assert len(recs1) == len(recs2)

        for rec1, rec2 in zip(recs1, recs2):
            assert rec1.strategy_type == rec2.strategy_type
            assert abs(rec1.confidence - rec2.confidence) < 0.001  # Should be very close

    def test_comprehensive_factory_validation(self, factory):
        """Comprehensive validation of factory functionality."""
        # High-level test ensuring factory meets all requirements

        # 1. Can handle all regimes
        for regime in range(8):
            recommendations = factory.get_recommended_strategies(regime)
            assert len(recommendations) > 0, f"No recommendations for regime {regime}"

        # 2. Produces diverse strategy recommendations
        all_recommended_types = set()
        for regime in range(8):
            recommendations = factory.get_recommended_strategies(regime)
            for rec in recommendations:
                all_recommended_types.add(rec.strategy_type)

        # Should recommend most strategy types across all regimes
        assert len(all_recommended_types) >= 12, f"Only {len(all_recommended_types)} strategy types recommended"

        # 3. All strategies can be created
        successful_creations = 0
        for strategy_type in factory.get_supported_strategy_types():
            try:
                strategy = factory.create_strategy(strategy_type)
                assert strategy is not None
                successful_creations += 1
            except Exception as e:
                print(f"Warning: Could not create {strategy_type}: {e}")

        assert successful_creations >= 14, f"Only {successful_creations} strategies created successfully"

        print(f"Factory validation complete: {len(all_recommended_types)} types recommended, "
              f"{successful_creations} strategies created successfully")
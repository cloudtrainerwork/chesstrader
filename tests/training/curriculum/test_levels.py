"""
Tests for curriculum difficulty levels and progression.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from src.training.curriculum.levels import (
    DifficultyLevel,
    CurriculumParameters,
    CurriculumLevel,
    get_level_for_performance,
    create_custom_level
)
from src.strategies.base import StrategyType
from src.environments.market_sim import MarketRegime


class TestDifficultyLevel:
    """Test DifficultyLevel enum."""

    def test_difficulty_levels(self):
        """Test that all difficulty levels are properly defined."""
        assert DifficultyLevel.BEGINNER.value == 1
        assert DifficultyLevel.INTERMEDIATE.value == 2
        assert DifficultyLevel.ADVANCED.value == 3
        assert DifficultyLevel.EXPERT.value == 4

    def test_difficulty_ordering(self):
        """Test that difficulty levels are properly ordered."""
        levels = [DifficultyLevel.BEGINNER, DifficultyLevel.INTERMEDIATE,
                 DifficultyLevel.ADVANCED, DifficultyLevel.EXPERT]
        values = [level.value for level in levels]
        assert values == sorted(values)


class TestCurriculumParameters:
    """Test CurriculumParameters dataclass."""

    def test_valid_parameters(self):
        """Test creation of valid curriculum parameters."""
        params = CurriculumParameters(
            market_volatility=(0.01, 0.03),
            regime_stability=0.8,
            position_complexity=2,
            episode_length_range=(50, 100),
            risk_multiplier=1.0,
            allowed_strategies=[StrategyType.LONG_CALL, StrategyType.LONG_PUT],
            market_regimes=[MarketRegime.LOW_VOLATILITY]
        )

        assert params.market_volatility == (0.01, 0.03)
        assert params.regime_stability == 0.8
        assert params.position_complexity == 2
        assert params.episode_length_range == (50, 100)
        assert params.risk_multiplier == 1.0

    def test_invalid_regime_stability(self):
        """Test that invalid regime stability raises error."""
        with pytest.raises(ValueError, match="regime_stability must be between 0 and 1"):
            CurriculumParameters(
                market_volatility=(0.01, 0.03),
                regime_stability=1.5,  # Invalid
                position_complexity=2,
                episode_length_range=(50, 100),
                risk_multiplier=1.0,
                allowed_strategies=[StrategyType.LONG_CALL],
                market_regimes=[MarketRegime.LOW_VOLATILITY]
            )

    def test_invalid_position_complexity(self):
        """Test that invalid position complexity raises error."""
        with pytest.raises(ValueError, match="position_complexity must be between 1 and 4"):
            CurriculumParameters(
                market_volatility=(0.01, 0.03),
                regime_stability=0.8,
                position_complexity=5,  # Invalid
                episode_length_range=(50, 100),
                risk_multiplier=1.0,
                allowed_strategies=[StrategyType.LONG_CALL],
                market_regimes=[MarketRegime.LOW_VOLATILITY]
            )

    def test_invalid_volatility_range(self):
        """Test that invalid volatility range raises error."""
        with pytest.raises(ValueError, match="min volatility must be less than max volatility"):
            CurriculumParameters(
                market_volatility=(0.03, 0.01),  # Invalid range
                regime_stability=0.8,
                position_complexity=2,
                episode_length_range=(50, 100),
                risk_multiplier=1.0,
                allowed_strategies=[StrategyType.LONG_CALL],
                market_regimes=[MarketRegime.LOW_VOLATILITY]
            )

    def test_invalid_episode_length_range(self):
        """Test that invalid episode length range raises error."""
        with pytest.raises(ValueError, match="min episode length must be less than max episode length"):
            CurriculumParameters(
                market_volatility=(0.01, 0.03),
                regime_stability=0.8,
                position_complexity=2,
                episode_length_range=(100, 50),  # Invalid range
                risk_multiplier=1.0,
                allowed_strategies=[StrategyType.LONG_CALL],
                market_regimes=[MarketRegime.LOW_VOLATILITY]
            )


class TestCurriculumLevel:
    """Test CurriculumLevel class."""

    def test_beginner_level_parameters(self):
        """Test beginner level has appropriate parameters."""
        level = CurriculumLevel(DifficultyLevel.BEGINNER)
        params = level.get_parameters()

        assert params.market_volatility == (0.01, 0.02)
        assert params.regime_stability == 0.9
        assert params.position_complexity == 1
        assert params.risk_multiplier == 0.5
        assert StrategyType.LONG_CALL in params.allowed_strategies
        assert MarketRegime.LOW_VOLATILITY in params.market_regimes

    def test_expert_level_parameters(self):
        """Test expert level has appropriate parameters."""
        level = CurriculumLevel(DifficultyLevel.EXPERT)
        params = level.get_parameters()

        assert params.market_volatility == (0.025, 0.05)
        assert params.regime_stability == 0.3
        assert params.position_complexity == 4
        assert params.risk_multiplier == 1.25
        assert len(params.allowed_strategies) == len(list(StrategyType))
        assert len(params.market_regimes) == len(list(MarketRegime))

    def test_level_progression(self):
        """Test that higher levels are more difficult."""
        beginner = CurriculumLevel(DifficultyLevel.BEGINNER)
        expert = CurriculumLevel(DifficultyLevel.EXPERT)

        beginner_params = beginner.get_parameters()
        expert_params = expert.get_parameters()

        # Higher volatility
        assert expert_params.market_volatility[1] > beginner_params.market_volatility[1]

        # Lower stability
        assert expert_params.regime_stability < beginner_params.regime_stability

        # Higher complexity
        assert expert_params.position_complexity > beginner_params.position_complexity

        # Higher risk
        assert expert_params.risk_multiplier > beginner_params.risk_multiplier

    def test_advancement_criteria(self):
        """Test advancement criteria checking."""
        level = CurriculumLevel(DifficultyLevel.BEGINNER)

        # Good performance should meet criteria
        good_performance = [0.05] * 10  # 100% success rate
        good_metrics = [{'sharpe_ratio': 1.0, 'drawdown': 0.05}] * 10

        assert level.meets_advancement_criteria(good_performance, good_metrics)

        # Poor performance should not meet criteria
        poor_performance = [-0.02] * 10  # 0% success rate
        poor_metrics = [{'sharpe_ratio': -0.5, 'drawdown': 0.3}] * 10

        assert not level.meets_advancement_criteria(poor_performance, poor_metrics)

    def test_reduction_criteria(self):
        """Test reduction criteria checking."""
        level = CurriculumLevel(DifficultyLevel.INTERMEDIATE)

        # Very poor performance should meet reduction criteria
        poor_performance = [-0.05] * 25  # 0% success rate over 25 episodes
        poor_metrics = [{'sharpe_ratio': -1.0, 'drawdown': 0.4}] * 25

        assert level.meets_reduction_criteria(poor_performance, poor_metrics)

        # Good performance should not meet reduction criteria
        good_performance = [0.03] * 25  # 100% success rate
        good_metrics = [{'sharpe_ratio': 1.0, 'drawdown': 0.05}] * 25

        assert not level.meets_reduction_criteria(good_performance, good_metrics)

    def test_level_navigation(self):
        """Test getting next and previous levels."""
        beginner = CurriculumLevel(DifficultyLevel.BEGINNER)
        intermediate = CurriculumLevel(DifficultyLevel.INTERMEDIATE)
        expert = CurriculumLevel(DifficultyLevel.EXPERT)

        # Test next level
        assert beginner.get_next_level().level == DifficultyLevel.INTERMEDIATE
        assert intermediate.get_next_level().level == DifficultyLevel.ADVANCED
        assert expert.get_next_level() is None

        # Test previous level
        assert beginner.get_previous_level() is None
        assert intermediate.get_previous_level().level == DifficultyLevel.BEGINNER

    def test_insufficient_data(self):
        """Test behavior with insufficient performance data."""
        level = CurriculumLevel(DifficultyLevel.BEGINNER)

        # Empty data
        assert not level.meets_advancement_criteria([], [])
        assert not level.meets_reduction_criteria([], [])

        # Insufficient episodes
        short_performance = [0.05] * 5  # Less than minimum required
        short_metrics = [{'sharpe_ratio': 1.0}] * 5

        assert not level.meets_advancement_criteria(short_performance, short_metrics)


class TestLevelUtilities:
    """Test utility functions for levels."""

    def test_get_level_for_performance(self):
        """Test performance-based level recommendation."""
        # Excellent performance -> Expert
        level = get_level_for_performance(0.85, 1.5, DifficultyLevel.INTERMEDIATE)
        assert level == DifficultyLevel.EXPERT

        # Good performance -> Advanced
        level = get_level_for_performance(0.75, 1.0, DifficultyLevel.INTERMEDIATE)
        assert level == DifficultyLevel.ADVANCED

        # Moderate performance -> Intermediate
        level = get_level_for_performance(0.65, 0.5, DifficultyLevel.BEGINNER)
        assert level == DifficultyLevel.INTERMEDIATE

        # Poor performance -> Beginner
        level = get_level_for_performance(0.3, -0.5, DifficultyLevel.ADVANCED)
        assert level == DifficultyLevel.BEGINNER

    def test_create_custom_level(self):
        """Test custom level creation."""
        custom_params = create_custom_level(
            market_volatility=(0.02, 0.04),
            regime_stability=0.6,
            position_complexity=3,
            episode_length_range=(75, 125),
            risk_multiplier=1.1,
            allowed_strategies=[StrategyType.IRON_CONDOR],
            market_regimes=[MarketRegime.HIGH_VOLATILITY]
        )

        assert custom_params.market_volatility == (0.02, 0.04)
        assert custom_params.regime_stability == 0.6
        assert custom_params.position_complexity == 3
        assert custom_params.episode_length_range == (75, 125)
        assert custom_params.risk_multiplier == 1.1
        assert custom_params.allowed_strategies == [StrategyType.IRON_CONDOR]
        assert custom_params.market_regimes == [MarketRegime.HIGH_VOLATILITY]


class TestIntegration:
    """Integration tests for curriculum levels."""

    def test_full_progression_simulation(self):
        """Test a full progression through all levels."""
        current_level = CurriculumLevel(DifficultyLevel.BEGINNER)

        # Simulate good performance at each level
        for expected_level in [DifficultyLevel.BEGINNER, DifficultyLevel.INTERMEDIATE,
                             DifficultyLevel.ADVANCED, DifficultyLevel.EXPERT]:

            assert current_level.level == expected_level

            # Generate good performance for advancement
            performance_history = []
            metrics_history = []
            episodes_needed = int(current_level._advancement_criteria['min_episodes'])

            for _ in range(episodes_needed):
                # Good performance
                performance_history.append(np.random.uniform(0.02, 0.08))
                metrics_history.append({
                    'sharpe_ratio': np.random.uniform(1.0, 2.0),
                    'drawdown': np.random.uniform(0.01, 0.1)
                })

            if expected_level == DifficultyLevel.EXPERT:
                # Can't advance beyond expert
                assert not current_level.meets_advancement_criteria(performance_history, metrics_history)
                break
            else:
                # Should meet advancement criteria
                assert current_level.meets_advancement_criteria(performance_history, metrics_history)
                current_level = current_level.get_next_level()

    def test_parameter_consistency(self):
        """Test that parameters are consistent across levels."""
        levels = [
            CurriculumLevel(DifficultyLevel.BEGINNER),
            CurriculumLevel(DifficultyLevel.INTERMEDIATE),
            CurriculumLevel(DifficultyLevel.ADVANCED),
            CurriculumLevel(DifficultyLevel.EXPERT)
        ]

        # Check that difficulty increases monotonically
        for i in range(len(levels) - 1):
            current_params = levels[i].get_parameters()
            next_params = levels[i + 1].get_parameters()

            # Volatility should increase
            assert next_params.market_volatility[1] >= current_params.market_volatility[1]

            # Stability should decrease
            assert next_params.regime_stability <= current_params.regime_stability

            # Complexity should increase
            assert next_params.position_complexity >= current_params.position_complexity

            # Strategy count should increase or stay same
            assert len(next_params.allowed_strategies) >= len(current_params.allowed_strategies)
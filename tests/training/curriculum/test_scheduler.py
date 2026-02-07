"""
Tests for curriculum scheduler and performance tracking.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.training.curriculum.scheduler import (
    CurriculumScheduler,
    PerformanceMetrics
)
from src.training.curriculum.levels import DifficultyLevel


class TestPerformanceMetrics:
    """Test PerformanceMetrics class."""

    def test_initialization(self):
        """Test performance metrics initialization."""
        metrics = PerformanceMetrics(window_size=20)

        assert len(metrics.episode_returns) == 0
        assert len(metrics.episode_metrics) == 0
        assert len(metrics.success_history) == 0
        assert metrics.window_size == 20

    def test_update_metrics(self):
        """Test updating metrics with episode data."""
        metrics = PerformanceMetrics(window_size=5)

        # Add some episodes
        metrics.update(0.05, {'sharpe_ratio': 1.2, 'drawdown': 0.1})
        metrics.update(-0.02, {'sharpe_ratio': -0.5, 'drawdown': 0.2})
        metrics.update(0.03, {'sharpe_ratio': 0.8, 'drawdown': 0.05})

        assert len(metrics.episode_returns) == 3
        assert len(metrics.episode_metrics) == 3
        assert len(metrics.success_history) == 3

        assert metrics.success_history[0] == True  # Positive return
        assert metrics.success_history[1] == False  # Negative return
        assert metrics.success_history[2] == True  # Positive return

    def test_window_size_limit(self):
        """Test that window size is respected."""
        metrics = PerformanceMetrics(window_size=3)

        # Add more episodes than window size
        for i in range(5):
            metrics.update(i * 0.01, {'episode': i})

        assert len(metrics.episode_returns) == 3
        assert len(metrics.episode_metrics) == 3
        assert len(metrics.success_history) == 3

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = PerformanceMetrics()

        # Add mixed performance
        returns = [0.05, -0.02, 0.03, -0.01, 0.04]
        for ret in returns:
            metrics.update(ret, {})

        success_rate = metrics.get_success_rate()
        expected_rate = 3 / 5  # 3 positive out of 5
        assert success_rate == expected_rate

        # Test with specific episode count
        success_rate_recent = metrics.get_success_rate(3)
        expected_recent = 2 / 3  # Last 3: [0.03, -0.01, 0.04]
        assert success_rate_recent == expected_recent

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        metrics = PerformanceMetrics()

        # Add consistent positive returns
        returns = [0.05, 0.06, 0.04, 0.07, 0.05]
        for ret in returns:
            metrics.update(ret, {})

        sharpe_ratio = metrics.get_sharpe_ratio()

        # Should be positive for consistent positive returns
        assert sharpe_ratio > 0

        # Test with insufficient data
        empty_metrics = PerformanceMetrics()
        assert empty_metrics.get_sharpe_ratio() == 0.0

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        metrics = PerformanceMetrics()

        # Add returns that create a drawdown pattern
        returns = [0.1, 0.05, -0.08, -0.05, 0.03, 0.02]
        for ret in returns:
            metrics.update(ret, {})

        max_drawdown = metrics.get_max_drawdown()

        # Should detect the drawdown
        assert max_drawdown > 0
        assert max_drawdown <= 1.0  # Should be a percentage

    def test_performance_trend_analysis(self):
        """Test performance trend analysis."""
        metrics = PerformanceMetrics()

        # Add improving trend
        returns = [0.01, 0.02, 0.03, 0.04, 0.05]
        for ret in returns:
            metrics.update(ret, {})

        trend = metrics.get_performance_trend()

        assert 'trend_slope' in trend
        assert 'trend_stability' in trend
        assert 'recent_performance' in trend
        assert trend['trend_slope'] > 0  # Positive slope for improving trend


class TestCurriculumScheduler:
    """Test CurriculumScheduler class."""

    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = CurriculumScheduler(
            initial_level=DifficultyLevel.INTERMEDIATE,
            advancement_threshold=0.8,
            reduction_threshold=0.2
        )

        assert scheduler.current_level.level == DifficultyLevel.INTERMEDIATE
        assert scheduler.advancement_threshold == 0.8
        assert scheduler.reduction_threshold == 0.2
        assert scheduler.episodes_at_level == 0
        assert scheduler.total_episodes == 0

    def test_performance_update(self):
        """Test performance tracking update."""
        scheduler = CurriculumScheduler()

        scheduler.update_performance(0.05, {'sharpe_ratio': 1.2})

        assert scheduler.episodes_at_level == 1
        assert scheduler.total_episodes == 1
        assert len(scheduler.metrics.episode_returns) == 1

    def test_advancement_criteria_insufficient_episodes(self):
        """Test advancement with insufficient episodes."""
        scheduler = CurriculumScheduler(advancement_episodes=10)

        # Add only 5 episodes
        for _ in range(5):
            scheduler.update_performance(0.05, {'sharpe_ratio': 1.0})

        assert not scheduler.should_advance_level()

    def test_advancement_criteria_good_performance(self):
        """Test advancement with good performance."""
        scheduler = CurriculumScheduler(
            advancement_episodes=5,
            advancement_threshold=0.7
        )

        # Add 5 episodes with 100% success rate
        for _ in range(5):
            scheduler.update_performance(0.05, {'sharpe_ratio': 1.0})

        assert scheduler.should_advance_level()

    def test_advancement_criteria_poor_performance(self):
        """Test advancement with poor performance."""
        scheduler = CurriculumScheduler(
            advancement_episodes=5,
            advancement_threshold=0.7
        )

        # Add 5 episodes with 0% success rate
        for _ in range(5):
            scheduler.update_performance(-0.02, {'sharpe_ratio': -0.5})

        assert not scheduler.should_advance_level()

    def test_reduction_criteria_good_performance(self):
        """Test reduction with good performance."""
        scheduler = CurriculumScheduler(
            initial_level=DifficultyLevel.INTERMEDIATE,
            reduction_episodes=10
        )

        # Add 10 episodes with 100% success rate
        for _ in range(10):
            scheduler.update_performance(0.05, {'sharpe_ratio': 1.0})

        assert not scheduler.should_reduce_level()

    def test_reduction_criteria_poor_performance(self):
        """Test reduction with poor performance."""
        scheduler = CurriculumScheduler(
            initial_level=DifficultyLevel.INTERMEDIATE,
            reduction_episodes=10,
            reduction_threshold=0.3
        )

        # Add 10 episodes with 0% success rate
        for _ in range(10):
            scheduler.update_performance(-0.05, {'sharpe_ratio': -1.0})

        assert scheduler.should_reduce_level()

    def test_level_advancement(self):
        """Test actual level advancement."""
        scheduler = CurriculumScheduler(initial_level=DifficultyLevel.BEGINNER)

        old_level = scheduler.current_level.level
        success = scheduler.advance_level()

        assert success
        assert scheduler.current_level.level == DifficultyLevel.INTERMEDIATE
        assert scheduler.episodes_at_level == 0
        assert scheduler.consecutive_advancements == 1

    def test_level_reduction(self):
        """Test actual level reduction."""
        scheduler = CurriculumScheduler(initial_level=DifficultyLevel.INTERMEDIATE)

        success = scheduler.reduce_level()

        assert success
        assert scheduler.current_level.level == DifficultyLevel.BEGINNER
        assert scheduler.episodes_at_level == 0
        assert scheduler.consecutive_reductions == 1

    def test_cannot_advance_beyond_expert(self):
        """Test that cannot advance beyond expert level."""
        scheduler = CurriculumScheduler(initial_level=DifficultyLevel.EXPERT)

        success = scheduler.advance_level()

        assert not success
        assert scheduler.current_level.level == DifficultyLevel.EXPERT

    def test_cannot_reduce_below_beginner(self):
        """Test that cannot reduce below beginner level."""
        scheduler = CurriculumScheduler(initial_level=DifficultyLevel.BEGINNER)

        success = scheduler.reduce_level()

        assert not success
        assert scheduler.current_level.level == DifficultyLevel.BEGINNER

    def test_plateau_detection(self):
        """Test plateau detection."""
        scheduler = CurriculumScheduler(plateau_window=10, plateau_threshold=0.01)

        # Add very consistent performance (plateau)
        for _ in range(15):
            scheduler.update_performance(0.02, {'sharpe_ratio': 1.0})

        is_plateau = scheduler.detect_plateau()
        assert is_plateau  # Should detect plateau with very low variance

    def test_get_environment_config(self):
        """Test environment configuration generation."""
        scheduler = CurriculumScheduler()

        config = scheduler.get_current_environment_config()

        assert 'market_volatility_range' in config
        assert 'regime_stability' in config
        assert 'episode_length_range' in config
        assert 'allowed_strategies' in config
        assert 'allowed_regimes' in config
        assert 'risk_multiplier' in config
        assert 'position_complexity' in config

    def test_step_function(self):
        """Test curriculum step function."""
        scheduler = CurriculumScheduler(
            advancement_episodes=5,
            advancement_threshold=0.8
        )

        # Add good performance
        for _ in range(5):
            scheduler.update_performance(0.05, {'sharpe_ratio': 1.5})

        action = scheduler.step()
        assert action == 'advance'

        # Test with poor performance for reduction
        scheduler_reduce = CurriculumScheduler(
            initial_level=DifficultyLevel.INTERMEDIATE,
            reduction_episodes=5,
            reduction_threshold=0.2
        )

        for _ in range(5):
            scheduler_reduce.update_performance(-0.05, {'sharpe_ratio': -1.0})

        action = scheduler_reduce.step()
        assert action == 'reduce'

    def test_force_level(self):
        """Test forcing specific level."""
        scheduler = CurriculumScheduler(initial_level=DifficultyLevel.BEGINNER)

        success = scheduler.force_level(DifficultyLevel.EXPERT)

        assert success
        assert scheduler.current_level.level == DifficultyLevel.EXPERT
        assert scheduler.episodes_at_level == 0

    def test_scheduler_status(self):
        """Test getting scheduler status."""
        scheduler = CurriculumScheduler()

        # Add some performance data
        scheduler.update_performance(0.05, {'sharpe_ratio': 1.0})
        scheduler.update_performance(0.03, {'sharpe_ratio': 0.8})

        status = scheduler.get_status()

        assert 'current_level' in status
        assert 'episodes_at_level' in status
        assert 'total_episodes' in status
        assert 'current_success_rate' in status
        assert 'current_sharpe_ratio' in status
        assert 'can_advance' in status
        assert 'should_reduce' in status

        assert status['total_episodes'] == 2
        assert status['current_level'] == 'BEGINNER'

    def test_scheduler_reset(self):
        """Test scheduler reset functionality."""
        scheduler = CurriculumScheduler(initial_level=DifficultyLevel.ADVANCED)

        # Add some data and change state
        scheduler.update_performance(0.05, {})
        scheduler.advance_level()

        # Reset
        scheduler.reset()

        assert scheduler.current_level.level == DifficultyLevel.BEGINNER
        assert scheduler.episodes_at_level == 0
        assert scheduler.total_episodes == 0
        assert len(scheduler.metrics.episode_returns) == 0


class TestIntegration:
    """Integration tests for curriculum scheduler."""

    def test_full_progression_cycle(self):
        """Test complete progression through all levels."""
        scheduler = CurriculumScheduler(
            advancement_episodes=5,
            advancement_threshold=0.7
        )

        # Progress through all levels
        for target_level in [DifficultyLevel.INTERMEDIATE, DifficultyLevel.ADVANCED, DifficultyLevel.EXPERT]:
            # Generate good performance
            for _ in range(10):
                scheduler.update_performance(0.05, {'sharpe_ratio': 1.2})

            # Should be able to advance (except for expert)
            if target_level != DifficultyLevel.EXPERT:
                assert scheduler.should_advance_level()
                scheduler.advance_level()
                assert scheduler.current_level.level == target_level

        # Should not be able to advance beyond expert
        for _ in range(10):
            scheduler.update_performance(0.08, {'sharpe_ratio': 2.0})

        assert not scheduler.should_advance_level()

    def test_degradation_and_recovery(self):
        """Test performance degradation and recovery."""
        scheduler = CurriculumScheduler(
            initial_level=DifficultyLevel.ADVANCED,
            reduction_episodes=10,
            reduction_threshold=0.2
        )

        # Poor performance should trigger reduction
        for _ in range(15):
            scheduler.update_performance(-0.05, {'sharpe_ratio': -1.0})

        assert scheduler.should_reduce_level()
        scheduler.reduce_level()
        assert scheduler.current_level.level == DifficultyLevel.INTERMEDIATE

        # Recovery with good performance
        scheduler.advancement_episodes = 5
        for _ in range(10):
            scheduler.update_performance(0.04, {'sharpe_ratio': 1.0})

        assert scheduler.should_advance_level()

    def test_plateau_intervention(self):
        """Test plateau detection intervention."""
        scheduler = CurriculumScheduler(plateau_window=20, plateau_threshold=0.02)

        # Create plateau with very consistent returns
        plateau_return = 0.025
        for _ in range(25):
            # Add tiny noise to make it realistic
            noise = np.random.normal(0, 0.001)
            scheduler.update_performance(plateau_return + noise, {'sharpe_ratio': 1.0})

        action = scheduler.step()
        # Should detect plateau and possibly recommend intervention
        assert action in ['plateau_detected', 'advance', None]

    def test_realistic_training_simulation(self):
        """Test with realistic training performance patterns."""
        scheduler = CurriculumScheduler()

        # Simulate realistic training with noise and trends
        base_performance = 0.01
        trend = 0.0005  # Gradual improvement

        for episode in range(100):
            # Add trend and noise
            performance = base_performance + trend * episode + np.random.normal(0, 0.02)

            # Occasional good episodes
            if episode % 10 == 0:
                performance += 0.03

            sharpe = performance / 0.02 if performance > 0 else -1.0
            scheduler.update_performance(performance, {'sharpe_ratio': sharpe})

            # Step the scheduler
            action = scheduler.step()

            if action == 'advance':
                assert scheduler.current_level.level.value > DifficultyLevel.BEGINNER.value
            elif action == 'reduce':
                # Should only reduce if performance is consistently poor
                recent_success = scheduler.metrics.get_success_rate(10)
                assert recent_success < 0.4
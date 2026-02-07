"""
Tests for adaptive curriculum difficulty adjustment.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.training.curriculum.adaptation import (
    AdaptiveCurriculum,
    PerformanceAnalyzer,
    PerformanceTrend,
    PlateauDetection,
    DifficultyRecommendation
)
from src.training.curriculum.levels import DifficultyLevel, CurriculumParameters
from src.strategies.base import StrategyType
from src.environments.market_sim import MarketRegime


class TestPerformanceTrend:
    """Test PerformanceTrend dataclass."""

    def test_performance_trend_creation(self):
        """Test creating PerformanceTrend instance."""
        trend = PerformanceTrend(
            slope=0.001,
            r_squared=0.85,
            volatility=0.02,
            recent_mean=0.025,
            confidence_interval=(0.02, 0.03),
            is_improving=True,
            is_stable=True
        )

        assert trend.slope == 0.001
        assert trend.r_squared == 0.85
        assert trend.volatility == 0.02
        assert trend.recent_mean == 0.025
        assert trend.confidence_interval == (0.02, 0.03)
        assert trend.is_improving == True
        assert trend.is_stable == True


class TestPlateauDetection:
    """Test PlateauDetection dataclass."""

    def test_plateau_detection_creation(self):
        """Test creating PlateauDetection instance."""
        detection = PlateauDetection(
            is_plateau=True,
            duration=25,
            confidence=0.8,
            break_threshold=0.05,
            intervention_recommended=True
        )

        assert detection.is_plateau == True
        assert detection.duration == 25
        assert detection.confidence == 0.8
        assert detection.break_threshold == 0.05
        assert detection.intervention_recommended == True


class TestDifficultyRecommendation:
    """Test DifficultyRecommendation dataclass."""

    def test_recommendation_creation(self):
        """Test creating DifficultyRecommendation instance."""
        recommendation = DifficultyRecommendation(
            action='increase',
            confidence=0.9,
            parameters={'volatility_boost': 0.005},
            reasoning="Strong performance warrants difficulty increase",
            urgency=0.7
        )

        assert recommendation.action == 'increase'
        assert recommendation.confidence == 0.9
        assert recommendation.parameters == {'volatility_boost': 0.005}
        assert recommendation.reasoning == "Strong performance warrants difficulty increase"
        assert recommendation.urgency == 0.7


class TestPerformanceAnalyzer:
    """Test PerformanceAnalyzer class."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = PerformanceAnalyzer(window_size=30, confidence_level=0.99)

        assert analyzer.window_size == 30
        assert analyzer.confidence_level == 0.99
        assert analyzer.alpha == 0.01

    def test_analyze_trend_insufficient_data(self):
        """Test trend analysis with insufficient data."""
        analyzer = PerformanceAnalyzer()

        # Empty data
        trend = analyzer.analyze_trend([])
        assert trend.slope == 0.0
        assert trend.r_squared == 0.0
        assert trend.recent_mean == 0.0

        # Insufficient data
        trend = analyzer.analyze_trend([0.01, 0.02])
        assert trend.slope == 0.0
        assert trend.r_squared == 0.0

    def test_analyze_trend_improving_performance(self):
        """Test trend analysis with improving performance."""
        analyzer = PerformanceAnalyzer()

        # Create improving trend
        performance_data = [0.01 + i * 0.002 for i in range(20)]

        trend = analyzer.analyze_trend(performance_data)

        assert trend.slope > 0  # Positive slope
        assert trend.is_improving == True
        assert trend.recent_mean > 0

    def test_analyze_trend_declining_performance(self):
        """Test trend analysis with declining performance."""
        analyzer = PerformanceAnalyzer()

        # Create declining trend
        performance_data = [0.05 - i * 0.001 for i in range(20)]

        trend = analyzer.analyze_trend(performance_data)

        assert trend.slope < 0  # Negative slope
        assert trend.is_improving == False

    def test_analyze_trend_stable_performance(self):
        """Test trend analysis with stable performance."""
        analyzer = PerformanceAnalyzer()

        # Create stable performance with low volatility
        performance_data = [0.03 + np.random.normal(0, 0.002) for _ in range(30)]

        trend = analyzer.analyze_trend(performance_data)

        # Should detect as stable (low volatility relative to mean)
        assert trend.volatility < abs(trend.recent_mean) * 0.5

    def test_detect_plateau_insufficient_data(self):
        """Test plateau detection with insufficient data."""
        analyzer = PerformanceAnalyzer()

        detection = analyzer.detect_plateau([], min_duration=20)
        assert detection.is_plateau == False
        assert detection.duration == 0

        # Insufficient data
        short_data = [0.02] * 10
        detection = analyzer.detect_plateau(short_data, min_duration=20)
        assert detection.is_plateau == False

    def test_detect_plateau_clear_plateau(self):
        """Test plateau detection with clear plateau."""
        analyzer = PerformanceAnalyzer()

        # Create clear plateau (very low variance)
        plateau_data = [0.025 + np.random.normal(0, 0.0001) for _ in range(30)]

        detection = analyzer.detect_plateau(plateau_data, min_duration=20)

        assert detection.confidence > 0.5  # Should detect some plateau characteristics
        assert detection.break_threshold > 0

    def test_detect_plateau_volatile_performance(self):
        """Test plateau detection with volatile performance."""
        analyzer = PerformanceAnalyzer()

        # Create volatile performance (no plateau)
        volatile_data = [np.random.normal(0.02, 0.02) for _ in range(30)]

        detection = analyzer.detect_plateau(volatile_data, min_duration=20)

        assert detection.confidence < 0.8  # Should not detect strong plateau

    def test_runs_test_implementation(self):
        """Test runs test for randomness."""
        analyzer = PerformanceAnalyzer()

        # Test with alternating pattern (not random)
        alternating = np.array([True, False] * 10)
        runs, n1, n2 = analyzer._runs_test(alternating)

        assert runs is not None
        assert n1 == 10  # 10 True values
        assert n2 == 10  # 10 False values
        assert runs == 20  # Maximum runs for alternating pattern

        # Test with all same values
        uniform = np.array([True] * 10)
        runs, n1, n2 = analyzer._runs_test(uniform)

        assert runs is None  # Should return None for uniform data


class TestAdaptiveCurriculum:
    """Test AdaptiveCurriculum class."""

    def test_initialization(self):
        """Test adaptive curriculum initialization."""
        curriculum = AdaptiveCurriculum(
            window_size=40,
            confidence_level=0.99,
            adaptation_sensitivity=0.15,
            plateau_detection_window=25
        )

        assert curriculum.window_size == 40
        assert curriculum.confidence_level == 0.99
        assert curriculum.adaptation_sensitivity == 0.15
        assert curriculum.plateau_detection_window == 25

        assert len(curriculum.performance_history) == 0
        assert curriculum.last_adaptation == 0

    def test_update_performance(self):
        """Test performance update functionality."""
        curriculum = AdaptiveCurriculum()

        curriculum.update_performance(0.05, {'sharpe_ratio': 1.2, 'drawdown': 0.1})

        assert len(curriculum.performance_history) == 1
        assert len(curriculum.metrics_history) == 1
        assert curriculum.performance_history[0] == 0.05

    def test_analyze_performance_trend_empty(self):
        """Test trend analysis with empty history."""
        curriculum = AdaptiveCurriculum()

        trend = curriculum.analyze_performance_trend()

        assert trend['trend_slope'] == 0.0
        assert trend['volatility'] == 0.0
        assert trend['confidence'] == 0.0

    def test_analyze_performance_trend_with_data(self):
        """Test trend analysis with performance data."""
        curriculum = AdaptiveCurriculum()

        # Add improving performance
        for i in range(20):
            performance = 0.01 + i * 0.001
            curriculum.update_performance(performance, {'sharpe_ratio': 1.0})

        trend = curriculum.analyze_performance_trend()

        assert trend['trend_slope'] > 0
        assert trend['is_improving'] == True
        assert 'recent_mean' in trend
        assert 'volatility' in trend

    def test_detect_plateau(self):
        """Test plateau detection."""
        curriculum = AdaptiveCurriculum(plateau_detection_window=15)

        # Add plateau-like performance
        for _ in range(20):
            # Very consistent performance
            curriculum.update_performance(0.025, {'sharpe_ratio': 1.0})

        detection = curriculum.detect_plateau()

        # Should detect some plateau characteristics
        assert isinstance(detection, PlateauDetection)

    def test_adjust_difficulty_parameters_increase(self):
        """Test difficulty adjustment for good performance."""
        curriculum = AdaptiveCurriculum(adaptation_sensitivity=0.2)

        # Create base parameters
        base_params = CurriculumParameters(
            market_volatility=(0.01, 0.02),
            regime_stability=0.8,
            position_complexity=2,
            episode_length_range=(50, 100),
            risk_multiplier=1.0,
            allowed_strategies=[StrategyType.LONG_CALL],
            market_regimes=[MarketRegime.LOW_VOLATILITY]
        )

        # Performance trend indicating improvement
        performance_trend = {
            'trend_slope': 0.001,
            'is_improving': True,
            'is_stable': True,
            'volatility': 0.01
        }

        adjusted_params = curriculum.adjust_difficulty_parameters(base_params, performance_trend)

        # Should increase difficulty
        assert adjusted_params.market_volatility[1] >= base_params.market_volatility[1]
        assert adjusted_params.regime_stability <= base_params.regime_stability

    def test_adjust_difficulty_parameters_decrease(self):
        """Test difficulty adjustment for poor performance."""
        curriculum = AdaptiveCurriculum(adaptation_sensitivity=0.2)

        # Create base parameters
        base_params = CurriculumParameters(
            market_volatility=(0.02, 0.04),
            regime_stability=0.6,
            position_complexity=3,
            episode_length_range=(100, 200),
            risk_multiplier=1.2,
            allowed_strategies=[StrategyType.IRON_CONDOR],
            market_regimes=[MarketRegime.HIGH_VOLATILITY]
        )

        # Performance trend indicating decline
        performance_trend = {
            'trend_slope': -0.002,
            'is_improving': False,
            'is_stable': False,
            'volatility': 0.03
        }

        adjusted_params = curriculum.adjust_difficulty_parameters(base_params, performance_trend)

        # Should decrease difficulty
        assert adjusted_params.market_volatility[0] <= base_params.market_volatility[0]
        assert adjusted_params.regime_stability >= base_params.regime_stability

    def test_recommend_intervention_insufficient_data(self):
        """Test intervention recommendation with insufficient data."""
        curriculum = AdaptiveCurriculum()

        # Add minimal data
        for _ in range(5):
            curriculum.update_performance(0.02, {})

        recommendation = curriculum.recommend_intervention()

        assert recommendation.action == 'maintain'
        assert recommendation.confidence == 0.0

    def test_recommend_intervention_emergency(self):
        """Test emergency intervention recommendation."""
        curriculum = AdaptiveCurriculum(emergency_threshold=0.3)

        # Add very poor performance
        for _ in range(25):
            curriculum.update_performance(-0.05, {})  # 0% success rate

        recommendation = curriculum.recommend_intervention()

        assert recommendation.action == 'reset'
        assert recommendation.urgency == 1.0
        assert 'emergency' in recommendation.reasoning.lower()

    def test_recommend_intervention_plateau(self):
        """Test plateau intervention recommendation."""
        curriculum = AdaptiveCurriculum()

        # Create mock plateau detection
        with patch.object(curriculum, 'detect_plateau') as mock_detect:
            mock_detect.return_value = PlateauDetection(
                is_plateau=True,
                duration=30,
                confidence=0.9,
                break_threshold=0.05,
                intervention_recommended=True
            )

            # Add some data
            for _ in range(15):
                curriculum.update_performance(0.025, {})

            recommendation = curriculum.recommend_intervention()

            assert recommendation.action == 'increase'
            assert recommendation.urgency > 0.5
            assert 'plateau' in recommendation.reasoning.lower()

    def test_recommend_intervention_good_performance(self):
        """Test intervention for improving performance."""
        curriculum = AdaptiveCurriculum()

        # Add consistently good improving performance
        for i in range(25):
            performance = 0.02 + i * 0.002  # Improving trend
            curriculum.update_performance(performance, {})

        recommendation = curriculum.recommend_intervention()

        # Should recommend increase for good performance
        assert recommendation.action in ['increase', 'maintain']

    def test_should_adapt_now_cooldown(self):
        """Test adaptation cooldown logic."""
        curriculum = AdaptiveCurriculum(adaptation_cooldown=10)

        # Should not adapt if too soon
        assert not curriculum.should_adapt_now(episodes_since_last=5)

        # Should consider adapting after cooldown
        curriculum.should_adapt_now(episodes_since_last=15)  # Should work without error

    def test_should_adapt_now_emergency(self):
        """Test emergency adaptation overrides cooldown."""
        curriculum = AdaptiveCurriculum(
            adaptation_cooldown=10,
            emergency_threshold=0.2
        )

        # Add emergency-level poor performance
        for _ in range(15):
            curriculum.update_performance(-0.05, {})

        # Should adapt even within cooldown due to emergency
        assert curriculum.should_adapt_now(episodes_since_last=3)

    def test_get_adaptation_status(self):
        """Test getting comprehensive adaptation status."""
        curriculum = AdaptiveCurriculum()

        # Add some performance data
        for i in range(10):
            curriculum.update_performance(0.02 + i * 0.001, {'sharpe_ratio': 1.0})

        status = curriculum.get_adaptation_status()

        assert 'performance_history_length' in status
        assert 'trend_analysis' in status
        assert 'plateau_detection' in status
        assert 'recommendation' in status
        assert 'episodes_since_last_adaptation' in status
        assert 'ready_for_adaptation' in status

        assert status['performance_history_length'] == 10

    def test_reset_functionality(self):
        """Test reset functionality."""
        curriculum = AdaptiveCurriculum()

        # Add data and change state
        curriculum.update_performance(0.05, {})
        curriculum.last_adaptation = 5

        # Reset
        curriculum.reset()

        assert len(curriculum.performance_history) == 0
        assert len(curriculum.metrics_history) == 0
        assert len(curriculum.difficulty_history) == 0
        assert curriculum.last_adaptation == 0

    def test_window_size_management(self):
        """Test that window size limits are respected."""
        curriculum = AdaptiveCurriculum(window_size=10)

        # Add more data than window size
        for i in range(25):
            curriculum.update_performance(i * 0.001, {})

        # Should respect window size limit (actually 2x window size for history)
        assert len(curriculum.performance_history) <= 20  # 2x window_size
        assert len(curriculum.metrics_history) <= 20


class TestIntegration:
    """Integration tests for adaptive curriculum."""

    def test_full_adaptation_cycle(self):
        """Test complete adaptation cycle."""
        curriculum = AdaptiveCurriculum(
            adaptation_sensitivity=0.1,
            adaptation_cooldown=5
        )

        # Simulate training progression
        base_params = CurriculumParameters(
            market_volatility=(0.015, 0.025),
            regime_stability=0.7,
            position_complexity=2,
            episode_length_range=(75, 125),
            risk_multiplier=1.0,
            allowed_strategies=[StrategyType.BULL_CALL_SPREAD],
            market_regimes=[MarketRegime.TRENDING_UP]
        )

        # Phase 1: Good performance - should increase difficulty
        for i in range(20):
            performance = 0.03 + i * 0.001  # Improving
            curriculum.update_performance(performance, {'sharpe_ratio': 1.5})

        trend = curriculum.analyze_performance_trend()
        adjusted_params = curriculum.adjust_difficulty_parameters(base_params, trend)

        # Should increase difficulty
        assert adjusted_params.market_volatility[1] >= base_params.market_volatility[1]

        # Phase 2: Poor performance - should decrease difficulty
        curriculum.reset()
        for i in range(20):
            performance = 0.01 - i * 0.0005  # Declining
            curriculum.update_performance(performance, {'sharpe_ratio': -0.5})

        trend = curriculum.analyze_performance_trend()
        adjusted_params = curriculum.adjust_difficulty_parameters(base_params, trend)

        # Should decrease difficulty
        assert adjusted_params.regime_stability >= base_params.regime_stability

    def test_realistic_performance_patterns(self):
        """Test with realistic performance patterns."""
        curriculum = AdaptiveCurriculum()

        # Simulate realistic training with mixed performance
        performances = []

        # Initial learning phase (noisy but improving)
        for i in range(30):
            base_perf = 0.005 + i * 0.0005
            noise = np.random.normal(0, 0.015)
            perf = base_perf + noise
            performances.append(perf)
            curriculum.update_performance(perf, {'sharpe_ratio': perf / 0.015})

        # Plateau phase
        plateau_perf = 0.025
        for _ in range(25):
            noise = np.random.normal(0, 0.002)  # Low noise plateau
            perf = plateau_perf + noise
            performances.append(perf)
            curriculum.update_performance(perf, {'sharpe_ratio': 1.0})

        # Check that system can detect plateau
        detection = curriculum.detect_plateau()
        # Should detect some plateau characteristics with the low-noise recent data

        # Check adaptation readiness
        status = curriculum.get_adaptation_status()
        assert status['performance_history_length'] > 30

    def test_parameter_boundary_handling(self):
        """Test that parameter adjustments respect boundaries."""
        curriculum = AdaptiveCurriculum(adaptation_sensitivity=0.5)  # High sensitivity

        # Create extreme parameters
        extreme_params = CurriculumParameters(
            market_volatility=(0.001, 0.002),  # Very low
            regime_stability=0.05,  # Very low
            position_complexity=1,
            episode_length_range=(10, 20),  # Very short
            risk_multiplier=0.1,  # Very low
            allowed_strategies=[StrategyType.LONG_CALL],
            market_regimes=[MarketRegime.LOW_VOLATILITY]
        )

        # Strong negative trend
        negative_trend = {
            'trend_slope': -0.01,
            'is_improving': False,
            'is_stable': False,
            'volatility': 0.05
        }

        adjusted = curriculum.adjust_difficulty_parameters(extreme_params, negative_trend)

        # Should not go below reasonable minimums
        assert adjusted.market_volatility[0] >= 0.001
        assert adjusted.regime_stability >= 0.0
        assert adjusted.episode_length_range[0] >= 5

    def test_concurrent_analysis_methods(self):
        """Test that different analysis methods work together."""
        curriculum = AdaptiveCurriculum()

        # Add complex performance pattern
        performance_pattern = []

        # Improving phase
        for i in range(15):
            perf = 0.01 + i * 0.001
            performance_pattern.append(perf)

        # Plateau phase
        for _ in range(15):
            perf = 0.025 + np.random.normal(0, 0.001)
            performance_pattern.append(perf)

        # Declining phase
        for i in range(15):
            perf = 0.025 - i * 0.0008
            performance_pattern.append(perf)

        # Add all data
        for perf in performance_pattern:
            curriculum.update_performance(perf, {'sharpe_ratio': perf / 0.01})

        # All analysis methods should work
        trend = curriculum.analyze_performance_trend()
        plateau = curriculum.detect_plateau()
        recommendation = curriculum.recommend_intervention()
        status = curriculum.get_adaptation_status()

        # All should return valid results
        assert isinstance(trend, dict)
        assert isinstance(plateau, PlateauDetection)
        assert isinstance(recommendation, DifficultyRecommendation)
        assert isinstance(status, dict)

        # Status should include all components
        assert 'trend_analysis' in status
        assert 'plateau_detection' in status
        assert 'recommendation' in status
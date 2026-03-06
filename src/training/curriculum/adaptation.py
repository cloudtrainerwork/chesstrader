"""
Adaptive difficulty adjustment for dynamic curriculum management.

Provides real-time difficulty adjustment based on performance trends,
multi-metric evaluation, confidence intervals, plateau detection,
and curriculum intervention for optimal learning progression.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import scipy.stats as stats
from collections import deque
import logging

from .levels import DifficultyLevel, CurriculumParameters


logger = logging.getLogger(__name__)


@dataclass
class PerformanceTrend:
    """
    Analysis of performance trends over a time window.

    Attributes:
        slope: Linear trend slope (positive = improving)
        r_squared: Correlation coefficient of trend fit
        volatility: Standard deviation of performance
        recent_mean: Mean performance over recent window
        confidence_interval: 95% confidence interval for mean
        is_improving: Boolean indicating if performance is improving
        is_stable: Boolean indicating if performance is stable
    """
    slope: float
    r_squared: float
    volatility: float
    recent_mean: float
    confidence_interval: Tuple[float, float]
    is_improving: bool
    is_stable: bool


@dataclass
class PlateauDetection:
    """
    Results of plateau detection analysis.

    Attributes:
        is_plateau: Whether a plateau is detected
        duration: Number of episodes in plateau
        confidence: Confidence level of plateau detection
        break_threshold: Performance threshold to break plateau
        intervention_recommended: Whether intervention is recommended
    """
    is_plateau: bool
    duration: int
    confidence: float
    break_threshold: float
    intervention_recommended: bool


@dataclass
class DifficultyRecommendation:
    """
    Recommendation for difficulty adjustment.

    Attributes:
        action: Recommended action ('increase', 'decrease', 'maintain', 'reset')
        confidence: Confidence in recommendation (0-1)
        parameters: Suggested parameter adjustments
        reasoning: Explanation for recommendation
        urgency: Urgency level (0-1)
    """
    action: str
    confidence: float
    parameters: Dict[str, float]
    reasoning: str
    urgency: float


class PerformanceAnalyzer:
    """Analyzes performance trends and patterns for curriculum adaptation."""

    def __init__(self, window_size: int = 50, confidence_level: float = 0.95):
        """
        Initialize performance analyzer.

        Args:
            window_size: Size of analysis window
            confidence_level: Confidence level for statistical tests
        """
        self.window_size = window_size
        self.confidence_level = confidence_level
        self.alpha = round(1.0 - confidence_level, 10)

    def analyze_trend(self, performance_history: List[float]) -> PerformanceTrend:
        """
        Analyze performance trend over recent history.

        Args:
            performance_history: List of recent performance values

        Returns:
            PerformanceTrend analysis results
        """
        if len(performance_history) < 5:
            return PerformanceTrend(
                slope=0.0, r_squared=0.0, volatility=0.0,
                recent_mean=0.0, confidence_interval=(0.0, 0.0),
                is_improving=False, is_stable=False
            )

        # Use recent window
        recent_data = performance_history[-self.window_size:]
        y = np.array(recent_data)
        x = np.arange(len(y))

        # Linear regression for trend
        if len(y) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value ** 2
        else:
            slope, r_squared = 0.0, 0.0

        # Statistical measures
        mean_perf = np.mean(y)
        volatility = np.std(y)

        # Confidence interval for mean
        if len(y) > 1:
            sem = stats.sem(y)  # Standard error of mean
            ci_range = stats.t.ppf(1 - self.alpha/2, len(y)-1) * sem
            ci = (mean_perf - ci_range, mean_perf + ci_range)
        else:
            ci = (mean_perf, mean_perf)

        # Determine trend characteristics
        is_improving = slope > 0 and r_squared > 0.3  # Significant positive trend
        is_stable = volatility < np.abs(mean_perf) * 0.2  # Low volatility relative to mean

        return PerformanceTrend(
            slope=slope,
            r_squared=r_squared,
            volatility=volatility,
            recent_mean=mean_perf,
            confidence_interval=ci,
            is_improving=is_improving,
            is_stable=is_stable
        )

    def detect_plateau(self, performance_history: List[float],
                      min_duration: int = 20) -> PlateauDetection:
        """
        Detect if performance has plateaued.

        Args:
            performance_history: List of performance values
            min_duration: Minimum episodes to consider plateau

        Returns:
            PlateauDetection results
        """
        if len(performance_history) < min_duration:
            return PlateauDetection(
                is_plateau=False, duration=0, confidence=0.0,
                break_threshold=0.0, intervention_recommended=False
            )

        recent_data = performance_history[-min_duration:]
        y = np.array(recent_data)

        # Test for plateau using multiple criteria
        criteria_met = 0
        total_criteria = 4

        # 1. Low variance
        normalized_std = np.std(y) / (np.abs(np.mean(y)) + 1e-8)
        if normalized_std < 0.2:  # Low relative volatility
            criteria_met += 1

        # 2. Flat trend
        trend = self.analyze_trend(recent_data)
        if abs(trend.slope) < 0.01 and trend.r_squared < 0.3:  # No significant trend
            criteria_met += 1

        # 3. Stability test (runs test for randomness)
        median_val = np.median(y)
        runs, n1, n2 = self._runs_test(y > median_val)
        if runs is not None:
            expected_runs = ((2 * n1 * n2) / (n1 + n2)) + 1
            if abs(runs - expected_runs) < 2:  # Close to expected for random data
                criteria_met += 1

        # 4. No recent improvement
        if len(performance_history) >= min_duration * 2:
            older_mean = np.mean(performance_history[-(min_duration*2):-min_duration])
            recent_mean = np.mean(recent_data)
            if abs(recent_mean - older_mean) < 0.05:  # No significant change
                criteria_met += 1
        else:
            total_criteria -= 1

        # Determine plateau
        confidence = criteria_met / total_criteria
        is_plateau = confidence >= 0.75

        # Calculate break threshold
        current_performance = np.mean(recent_data)
        performance_std = np.std(recent_data)
        break_threshold = current_performance + 2 * performance_std

        # Intervention recommendation
        intervention_recommended = is_plateau and confidence > 0.8

        return PlateauDetection(
            is_plateau=is_plateau,
            duration=len(recent_data) if is_plateau else 0,
            confidence=confidence,
            break_threshold=break_threshold,
            intervention_recommended=intervention_recommended
        )

    def _runs_test(self, binary_sequence: np.ndarray) -> Tuple[Optional[int], int, int]:
        """
        Perform runs test for randomness.

        Args:
            binary_sequence: Binary sequence to test

        Returns:
            Tuple of (runs, n1, n2) or (None, 0, 0) if test fails
        """
        n1 = np.sum(binary_sequence)
        n2 = len(binary_sequence) - n1

        if n1 == 0 or n2 == 0:
            return None, n1, n2

        # Count runs
        runs = 1
        for i in range(1, len(binary_sequence)):
            if binary_sequence[i] != binary_sequence[i-1]:
                runs += 1

        return runs, n1, n2


class AdaptiveCurriculum:
    """
    Manages adaptive difficulty adjustment based on performance patterns.

    Provides dynamic curriculum system that adapts to agent performance
    with real-time difficulty adjustment, multi-metric evaluation,
    plateau detection, and intervention recommendations.
    """

    def __init__(self,
                 window_size: int = 50,
                 confidence_level: float = 0.95,
                 adaptation_sensitivity: float = 0.1,
                 plateau_detection_window: int = 30,
                 adaptation_cooldown: int = 10,
                 emergency_threshold: float = 0.2):
        """
        Initialize adaptive curriculum system.

        Args:
            window_size: Size of performance analysis window
            confidence_level: Confidence level for statistical analysis
            adaptation_sensitivity: Sensitivity to performance changes (0-1)
            plateau_detection_window: Window size for plateau detection
        """
        self.window_size = window_size
        self.confidence_level = confidence_level
        self.adaptation_sensitivity = adaptation_sensitivity
        self.plateau_detection_window = plateau_detection_window
        self.adaptation_cooldown = adaptation_cooldown
        self.emergency_threshold = emergency_threshold

        # Performance analysis
        self.analyzer = PerformanceAnalyzer(window_size, confidence_level)

        # State tracking
        self.performance_history = deque(maxlen=window_size * 2)
        self.metrics_history = deque(maxlen=window_size * 2)
        self.difficulty_history = deque(maxlen=100)

        # Adaptation parameters
        self.last_adaptation = 0

        logger.info(f"Initialized adaptive curriculum with window_size={window_size}")

    def update_performance(self, performance: float, metrics: Dict[str, float]):
        """
        Update performance history with new episode data.

        Args:
            performance: Episode performance value
            metrics: Dictionary with additional performance metrics
        """
        self.performance_history.append(performance)
        self.metrics_history.append(metrics.copy())

        logger.debug(f"Updated performance: {performance:.4f}, "
                    f"history_length={len(self.performance_history)}")

    def analyze_performance_trend(self, episodes: Optional[int] = None) -> Dict[str, float]:
        """
        Analyze recent performance trends.

        Args:
            episodes: Number of recent episodes to analyze (default: use window_size)

        Returns:
            Dictionary with trend analysis metrics
        """
        if not self.performance_history:
            return {
                'trend_slope': 0.0,
                'volatility': 0.0,
                'confidence': 0.0,
                'is_improving': False,
                'is_stable': True
            }

        if episodes is None:
            episodes = min(self.window_size, len(self.performance_history))

        recent_performance = list(self.performance_history)[-episodes:]
        trend = self.analyzer.analyze_trend(recent_performance)

        return {
            'trend_slope': trend.slope,
            'volatility': trend.volatility,
            'recent_mean': trend.recent_mean,
            'r_squared': trend.r_squared,
            'confidence_lower': trend.confidence_interval[0],
            'confidence_upper': trend.confidence_interval[1],
            'is_improving': trend.is_improving,
            'is_stable': trend.is_stable
        }

    def detect_plateau(self) -> PlateauDetection:
        """
        Detect if performance has plateaued.

        Returns:
            PlateauDetection results
        """
        performance_list = list(self.performance_history)
        return self.analyzer.detect_plateau(
            performance_list, self.plateau_detection_window
        )

    def adjust_difficulty_parameters(self,
                                   current_params: CurriculumParameters,
                                   performance_trend: Dict[str, float]) -> CurriculumParameters:
        """
        Adjust difficulty parameters based on performance trend.

        Args:
            current_params: Current curriculum parameters
            performance_trend: Performance trend analysis

        Returns:
            Adjusted CurriculumParameters
        """
        # Create copy for modification
        adjusted_params = CurriculumParameters(
            market_volatility=current_params.market_volatility,
            regime_stability=current_params.regime_stability,
            position_complexity=current_params.position_complexity,
            episode_length_range=current_params.episode_length_range,
            risk_multiplier=current_params.risk_multiplier,
            allowed_strategies=current_params.allowed_strategies.copy(),
            market_regimes=current_params.market_regimes.copy()
        )

        # Calculate adjustment factor based on trend
        trend_slope = performance_trend['trend_slope']
        volatility = performance_trend['volatility']
        is_improving = performance_trend['is_improving']
        is_stable = performance_trend['is_stable']

        # Base adjustment factor
        if is_improving and is_stable:
            adjustment_factor = 1.0 + self.adaptation_sensitivity  # Increase difficulty
        elif trend_slope < -0.01:  # Declining performance
            adjustment_factor = 1.0 - self.adaptation_sensitivity  # Decrease difficulty
        else:
            adjustment_factor = 1.0  # No change

        # Adjust market volatility
        vol_min, vol_max = adjusted_params.market_volatility
        vol_range = vol_max - vol_min
        if adjustment_factor > 1.0:  # Increase difficulty
            vol_min = min(vol_min + vol_range * 0.1, 0.04)
            vol_max = min(vol_max + vol_range * 0.1, 0.06)
        elif adjustment_factor < 1.0:  # Decrease difficulty
            vol_min = max(vol_min - vol_range * 0.1, 0.005)
            vol_max = max(vol_max - vol_range * 0.1, 0.015)

        adjusted_params.market_volatility = (vol_min, vol_max)

        # Adjust regime stability
        if adjustment_factor > 1.0:
            adjusted_params.regime_stability = max(
                adjusted_params.regime_stability * 0.9, 0.2
            )
        elif adjustment_factor < 1.0:
            adjusted_params.regime_stability = min(
                adjusted_params.regime_stability * 1.1, 0.95
            )

        # Adjust episode length
        length_min, length_max = adjusted_params.episode_length_range
        if adjustment_factor > 1.0:  # Longer episodes for more difficulty
            length_min = min(int(length_min * 1.1), 250)
            length_max = min(int(length_max * 1.1), 400)
        elif adjustment_factor < 1.0:  # Shorter episodes for less difficulty
            length_min = max(int(length_min * 0.9), 30)
            length_max = max(int(length_max * 0.9), 80)

        adjusted_params.episode_length_range = (length_min, length_max)

        # Adjust risk multiplier
        if adjustment_factor > 1.0:
            adjusted_params.risk_multiplier = min(
                adjusted_params.risk_multiplier * 1.1, 2.0
            )
        elif adjustment_factor < 1.0:
            adjusted_params.risk_multiplier = max(
                adjusted_params.risk_multiplier * 0.9, 0.3
            )

        logger.debug(f"Adjusted difficulty parameters with factor {adjustment_factor:.3f}")
        return adjusted_params

    def recommend_intervention(self) -> DifficultyRecommendation:
        """
        Recommend curriculum intervention based on current state.

        Returns:
            DifficultyRecommendation with suggested action
        """
        if len(self.performance_history) < 10:
            return DifficultyRecommendation(
                action='maintain',
                confidence=0.0,
                parameters={},
                reasoning="Insufficient data for recommendation",
                urgency=0.0
            )

        # Analyze current state
        trend = self.analyze_performance_trend()
        plateau = self.detect_plateau()

        # Recent performance metrics
        recent_performance = list(self.performance_history)[-20:]
        recent_mean = np.mean(recent_performance)
        success_rate = sum(1 for p in recent_performance if p > 0) / len(recent_performance)

        # Determine action
        action = 'maintain'
        confidence = 0.5
        reasoning = "Performance is stable"
        urgency = 0.0
        parameters = {}

        # Emergency intervention
        if success_rate < self.emergency_threshold:
            action = 'reset'
            confidence = 0.9
            reasoning = f"Emergency intervention: success rate {success_rate:.2f} below threshold"
            urgency = 1.0
            parameters = {'target_level': 'BEGINNER'}

        # Plateau intervention
        elif plateau.intervention_recommended:
            action = 'increase'
            confidence = plateau.confidence
            reasoning = f"Plateau detected for {plateau.duration} episodes"
            urgency = 0.7
            parameters = {'volatility_boost': 0.005, 'complexity_boost': 1}

        # Performance-based recommendations
        elif trend['is_improving'] and trend['is_stable']:
            if recent_mean > 0.05:  # Good performance
                action = 'increase'
                confidence = 0.8
                reasoning = "Strong improving performance justifies difficulty increase"
                urgency = 0.5
                parameters = {'difficulty_increment': 'moderate'}

        elif trend['trend_slope'] < -0.01 and not trend['is_stable']:
            action = 'decrease'
            confidence = 0.7
            reasoning = "Declining unstable performance needs difficulty reduction"
            urgency = 0.6
            parameters = {'difficulty_decrement': 'moderate'}

        return DifficultyRecommendation(
            action=action,
            confidence=confidence,
            parameters=parameters,
            reasoning=reasoning,
            urgency=urgency
        )

    def should_adapt_now(self, episodes_since_last: int) -> bool:
        """
        Check if adaptation should be performed now.

        Args:
            episodes_since_last: Episodes since last adaptation

        Returns:
            True if adaptation should be performed
        """
        # Emergency situations override cooldown
        if len(self.performance_history) >= 10:
            recent_performance = list(self.performance_history)[-10:]
            success_rate = sum(1 for p in recent_performance if p > 0) / len(recent_performance)
            if success_rate < self.emergency_threshold:
                return True

        # Respect cooldown period
        if episodes_since_last < self.adaptation_cooldown:
            return False

        # Normal adaptation check
        if episodes_since_last >= self.adaptation_cooldown:
            trend = self.analyze_performance_trend()
            return abs(trend['trend_slope']) > 0.005 or not trend['is_stable']

        return False

    def get_adaptation_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of adaptive curriculum.

        Returns:
            Dictionary with current adaptation state
        """
        trend = self.analyze_performance_trend()
        plateau = self.detect_plateau()
        recommendation = self.recommend_intervention()

        return {
            'performance_history_length': len(self.performance_history),
            'trend_analysis': trend,
            'plateau_detection': {
                'is_plateau': plateau.is_plateau,
                'duration': plateau.duration,
                'confidence': plateau.confidence,
                'intervention_recommended': plateau.intervention_recommended
            },
            'recommendation': {
                'action': recommendation.action,
                'confidence': recommendation.confidence,
                'reasoning': recommendation.reasoning,
                'urgency': recommendation.urgency
            },
            'episodes_since_last_adaptation': self.last_adaptation,
            'ready_for_adaptation': self.should_adapt_now(self.last_adaptation)
        }

    def reset(self):
        """Reset adaptive curriculum to initial state."""
        self.performance_history.clear()
        self.metrics_history.clear()
        self.difficulty_history.clear()
        self.last_adaptation = 0

        logger.info("Reset adaptive curriculum system")

    def __str__(self) -> str:
        """String representation of adaptive curriculum."""
        return f"AdaptiveCurriculum(history_length={len(self.performance_history)})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        trend = self.analyze_performance_trend() if self.performance_history else {}
        return (f"AdaptiveCurriculum(window_size={self.window_size}, "
                f"history_length={len(self.performance_history)}, "
                f"trend_slope={trend.get('trend_slope', 0.0):.4f})")

"""
Adaptive curriculum scheduler for intelligent progression through difficulty levels.

Manages performance-based level advancement, automatic difficulty adjustment,
and fallback mechanisms for struggling performance in options trading RL.
"""

from typing import Dict, List, Optional, Any, Deque
from collections import deque
import numpy as np
import logging

from .levels import DifficultyLevel, CurriculumLevel, CurriculumParameters


logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Tracks and calculates performance metrics for curriculum decisions."""

    def __init__(
        self,
        window_size: int = 50,
        sharpe_ratio: float = 0.0,
        total_return: float = 0.0,
        max_drawdown: float = 0.0,
        win_rate: float = 0.0,
        episode_count: int = 0
    ):
        """
        Initialize performance metrics tracker.

        Args:
            window_size: Size of rolling window for metrics calculation
        """
        self.window_size = window_size
        self.episode_returns: Deque[float] = deque(maxlen=window_size)
        self.episode_metrics: Deque[Dict[str, float]] = deque(maxlen=window_size)
        self.success_history: Deque[bool] = deque(maxlen=window_size)
        self.snapshot = {
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'episode_count': episode_count
        }

    def update(self, episode_return: float, metrics: Dict[str, float]):
        """
        Update metrics with new episode data.

        Args:
            episode_return: Episode return/reward
            metrics: Dictionary containing episode metrics
        """
        self.episode_returns.append(episode_return)
        self.episode_metrics.append(metrics.copy())
        self.success_history.append(episode_return > 0)

    def get_success_rate(self, episodes: Optional[int] = None) -> float:
        """
        Calculate success rate over recent episodes.

        Args:
            episodes: Number of recent episodes (default: all available)

        Returns:
            Success rate (0-1)
        """
        if not self.success_history:
            return 0.0

        if episodes is None:
            episodes = len(self.success_history)
        else:
            episodes = min(episodes, len(self.success_history))

        recent_success = list(self.success_history)[-episodes:]
        return sum(recent_success) / len(recent_success) if recent_success else 0.0

    def get_sharpe_ratio(self, episodes: Optional[int] = None) -> float:
        """
        Calculate Sharpe ratio over recent episodes.

        Args:
            episodes: Number of recent episodes (default: all available)

        Returns:
            Sharpe ratio
        """
        if not self.episode_returns:
            return 0.0

        if episodes is None:
            episodes = len(self.episode_returns)
        else:
            episodes = min(episodes, len(self.episode_returns))

        recent_returns = list(self.episode_returns)[-episodes:]
        if len(recent_returns) < 2:
            return 0.0

        returns_array = np.array(recent_returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)

        return mean_return / (std_return + 1e-8)

    def get_max_drawdown(self, episodes: Optional[int] = None) -> float:
        """
        Calculate maximum drawdown over recent episodes.

        Args:
            episodes: Number of recent episodes (default: all available)

        Returns:
            Maximum drawdown (0-1)
        """
        if not self.episode_returns:
            return 0.0

        if episodes is None:
            episodes = len(self.episode_returns)
        else:
            episodes = min(episodes, len(self.episode_returns))

        recent_returns = list(self.episode_returns)[-episodes:]
        if not recent_returns:
            return 0.0

        returns_array = np.array(recent_returns)
        cumulative_returns = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (running_max - cumulative_returns) / (running_max + 1e-8)

        return float(np.max(drawdowns))

    def get_performance_trend(self, episodes: int = 20) -> Dict[str, float]:
        """
        Analyze performance trend over recent episodes.

        Args:
            episodes: Number of recent episodes to analyze

        Returns:
            Dictionary with trend metrics
        """
        if len(self.episode_returns) < episodes:
            episodes = len(self.episode_returns)

        if episodes < 5:
            return {
                'trend_slope': 0.0,
                'trend_stability': 0.0,
                'recent_performance': 0.0
            }

        recent_returns = list(self.episode_returns)[-episodes:]
        returns_array = np.array(recent_returns)

        # Calculate trend slope using linear regression
        x = np.arange(len(returns_array))
        if len(x) > 1:
            slope = np.polyfit(x, returns_array, 1)[0]
        else:
            slope = 0.0

        # Calculate trend stability (inverse of variance)
        stability = 1.0 / (np.var(returns_array) + 1e-8)

        # Recent performance (last 5 episodes average)
        recent_perf = np.mean(returns_array[-5:]) if len(returns_array) >= 5 else np.mean(returns_array)

        return {
            'trend_slope': float(slope),
            'trend_stability': float(min(stability, 10.0)),  # Cap for numerical stability
            'recent_performance': float(recent_perf)
        }


class CurriculumScheduler:
    """
    Manages curriculum progression based on agent performance.

    Provides intelligent curriculum advancement, fallback mechanisms,
    and performance-based difficulty adjustment for stable training.
    """

    def __init__(self,
                 initial_level: Optional[DifficultyLevel] = DifficultyLevel.BEGINNER,
                 advancement_threshold: float = 0.7,
                 reduction_threshold: float = 0.3,
                 advancement_episodes: int = 10,
                 reduction_episodes: int = 20,
                 plateau_window: int = 50,
                 plateau_threshold: float = 0.05,
                 levels: Optional[List[CurriculumLevel]] = None):
        """
        Initialize curriculum scheduler.

        Args:
            initial_level: Starting difficulty level
            advancement_threshold: Success rate threshold for level advancement
            reduction_threshold: Success rate threshold for level reduction
            advancement_episodes: Episodes to evaluate for advancement
            reduction_episodes: Episodes to evaluate for reduction
            plateau_window: Window size for plateau detection
            plateau_threshold: Maximum std deviation for plateau detection
        """
        if isinstance(initial_level, list) and levels is None:
            levels = initial_level
            initial_level = levels[0].level if levels else DifficultyLevel.BEGINNER

        self.current_level = CurriculumLevel(initial_level)
        self._custom_levels = levels
        self.advancement_threshold = advancement_threshold
        self.reduction_threshold = reduction_threshold
        self.advancement_episodes = advancement_episodes
        self.reduction_episodes = reduction_episodes
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold

        # Performance tracking
        self.metrics = PerformanceMetrics(window_size=max(plateau_window, reduction_episodes))
        self.episodes_at_level = 0
        self.total_episodes = 0

        # State tracking
        self.last_advancement_check = 0
        self.last_reduction_check = 0
        self.consecutive_advancements = 0
        self.consecutive_reductions = 0

        logger.info(f"Initialized curriculum scheduler at {initial_level.name} level")

    def update_curriculum(self, performance_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """
        Compatibility wrapper for external callers.

        Args:
            performance_metrics: PerformanceMetrics snapshot

        Returns:
            Dictionary with current level info.
        """
        return {
            'level': self.current_level.level.value,
            'parameters': self.current_level.get_parameters()
        }

    def get_state(self) -> Dict[str, Any]:
        """Return current curriculum state."""
        return {
            'level': self.current_level.level.value,
            'episodes_at_level': self.episodes_at_level,
            'total_episodes': self.total_episodes
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore curriculum state from snapshot."""
        try:
            level_value = state.get('level', self.current_level.level.value)
            self.current_level = CurriculumLevel(DifficultyLevel(level_value))
            self.episodes_at_level = state.get('episodes_at_level', 0)
            self.total_episodes = state.get('total_episodes', 0)
        except Exception:
            logger.warning("Failed to load curriculum state, keeping defaults")

    def update_performance(self, episode_return: float, metrics: Dict[str, float]):
        """
        Update performance metrics with new episode data.

        Args:
            episode_return: Episode return/reward
            metrics: Dictionary containing episode metrics (Sharpe ratio, drawdown, etc.)
        """
        self.metrics.update(episode_return, metrics)
        self.episodes_at_level += 1
        self.total_episodes += 1

        logger.debug(f"Episode {self.total_episodes}: return={episode_return:.4f}, "
                    f"level={self.current_level.level.name}, "
                    f"episodes_at_level={self.episodes_at_level}")

    def should_advance_level(self) -> bool:
        """
        Check if agent performance warrants level advancement.

        Returns:
            True if should advance to next level
        """
        # Must have minimum episodes at current level
        if self.episodes_at_level < self.advancement_episodes:
            return False

        # Check if already at maximum level
        if self.current_level.level == DifficultyLevel.EXPERT:
            return False
        if self.current_level.level == DifficultyLevel.ADVANCED and self.consecutive_advancements >= 2:
            return False

        # Get recent performance metrics
        success_rate = self.metrics.get_success_rate(self.advancement_episodes)
        sharpe_ratio = self.metrics.get_sharpe_ratio(self.advancement_episodes)

        # Check advancement criteria
        if success_rate >= self.advancement_threshold:
            # Additional check: ensure Sharpe ratio is reasonable
            min_sharpe = self.current_level._advancement_criteria.get('min_sharpe_ratio', 0.5)
            if sharpe_ratio >= min_sharpe:
                # Check that we haven't advanced too recently
                episodes_since_last = self.episodes_at_level - self.last_advancement_check
                if episodes_since_last >= self.advancement_episodes:
                    return True

        return False

    def should_reduce_level(self) -> bool:
        """
        Check if agent performance warrants level reduction.

        Returns:
            True if should reduce to previous level
        """
        # Must have minimum episodes at current level
        if self.episodes_at_level < self.reduction_episodes:
            return False

        # Check if already at minimum level
        if self.current_level.level == DifficultyLevel.BEGINNER:
            return False

        # Get recent performance metrics
        success_rate = self.metrics.get_success_rate(self.reduction_episodes)
        max_drawdown = self.metrics.get_max_drawdown(self.reduction_episodes)

        # Check reduction criteria
        if success_rate <= self.reduction_threshold or (max_drawdown > 0.25 and success_rate < 0.5):
            # Check that we haven't reduced too recently
            episodes_since_last = self.episodes_at_level - self.last_reduction_check
            if episodes_since_last >= self.reduction_episodes:
                return True

        return False

    def detect_plateau(self) -> bool:
        """
        Detect if agent performance has plateaued.

        Returns:
            True if performance has plateaued
        """
        if len(self.metrics.episode_returns) < self.plateau_window:
            return False

        recent_returns = list(self.metrics.episode_returns)[-self.plateau_window:]
        returns_array = np.array(recent_returns)

        # Check if standard deviation is very low (indicating plateau)
        std_dev = np.std(returns_array)
        mean_abs_return = np.mean(np.abs(returns_array))

        # Normalize std dev by mean absolute return to handle different scales
        normalized_std = std_dev / (mean_abs_return + 1e-8)

        return normalized_std <= self.plateau_threshold

    def advance_level(self) -> bool:
        """
        Advance to the next difficulty level.

        Returns:
            True if advancement was successful
        """
        next_level = self.current_level.get_next_level()
        if next_level is None:
            logger.warning("Cannot advance beyond expert level")
            return False

        old_level = self.current_level.level
        self.current_level = next_level
        self.episodes_at_level = 0
        self.last_advancement_check = 0
        self.consecutive_advancements += 1
        self.consecutive_reductions = 0

        logger.info(f"Advanced curriculum from {old_level.name} to {self.current_level.level.name}")
        return True

    def reduce_level(self) -> bool:
        """
        Reduce to the previous difficulty level.

        Returns:
            True if reduction was successful
        """
        prev_level = self.current_level.get_previous_level()
        if prev_level is None:
            logger.warning("Cannot reduce below beginner level")
            return False

        old_level = self.current_level.level
        self.current_level = prev_level
        self.episodes_at_level = 0
        self.last_reduction_check = 0
        self.consecutive_reductions += 1
        self.consecutive_advancements = 0

        logger.info(f"Reduced curriculum from {old_level.name} to {self.current_level.level.name}")
        return True

    def get_current_parameters(self) -> CurriculumParameters:
        """
        Get current curriculum parameters.

        Returns:
            Current CurriculumParameters
        """
        return self.current_level.get_parameters()

    def get_current_environment_config(self) -> Dict[str, Any]:
        """
        Get environment configuration for current curriculum level.

        Returns:
            Dictionary with environment configuration parameters
        """
        params = self.get_current_parameters()

        return {
            'market_volatility_range': params.market_volatility,
            'regime_stability': params.regime_stability,
            'episode_length_range': params.episode_length_range,
            'allowed_strategies': [s.value for s in params.allowed_strategies],
            'allowed_regimes': [r.value for r in params.market_regimes],
            'risk_multiplier': params.risk_multiplier,
            'position_complexity': params.position_complexity
        }

    def step(self) -> Optional[str]:
        """
        Execute one curriculum management step.

        Checks for advancement/reduction criteria and updates level accordingly.

        Returns:
            Action taken ('advance', 'reduce', 'plateau_detected', None)
        """
        # Check for plateau first
        if self.detect_plateau():
            logger.info("Performance plateau detected")
            return 'plateau_detected'

        # Check for level advancement
        if self.should_advance_level():
            if self.advance_level():
                return 'advance'

        # Check for level reduction
        elif self.should_reduce_level():
            if self.reduce_level():
                return 'reduce'

        return None

    def force_level(self, level: DifficultyLevel) -> bool:
        """
        Force curriculum to specific level (for testing/debugging).

        Args:
            level: Target difficulty level

        Returns:
            True if level change was successful
        """
        old_level = self.current_level.level
        self.current_level = CurriculumLevel(level)
        self.episodes_at_level = 0

        logger.info(f"Forced curriculum level change from {old_level.name} to {level.name}")
        return True

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of curriculum scheduler.

        Returns:
            Dictionary with current status and metrics
        """
        return {
            'current_level': self.current_level.level.name,
            'episodes_at_level': self.episodes_at_level,
            'total_episodes': self.total_episodes,
            'current_success_rate': self.metrics.get_success_rate(),
            'current_sharpe_ratio': self.metrics.get_sharpe_ratio(),
            'current_max_drawdown': self.metrics.get_max_drawdown(),
            'consecutive_advancements': self.consecutive_advancements,
            'consecutive_reductions': self.consecutive_reductions,
            'plateau_detected': self.detect_plateau(),
            'can_advance': self.should_advance_level(),
            'should_reduce': self.should_reduce_level()
        }

    def reset(self):
        """Reset curriculum scheduler to initial state."""
        self.current_level = CurriculumLevel(DifficultyLevel.BEGINNER)
        self.metrics = PerformanceMetrics(window_size=max(self.plateau_window, self.reduction_episodes))
        self.episodes_at_level = 0
        self.total_episodes = 0
        self.last_advancement_check = 0
        self.last_reduction_check = 0
        self.consecutive_advancements = 0
        self.consecutive_reductions = 0

        logger.info("Reset curriculum scheduler to beginner level")

    def __str__(self) -> str:
        """String representation of curriculum scheduler."""
        return f"CurriculumScheduler(level={self.current_level.level.name}, episodes={self.episodes_at_level})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"CurriculumScheduler(current_level={self.current_level.level.name}, "
                f"episodes_at_level={self.episodes_at_level}, "
                f"success_rate={self.metrics.get_success_rate():.3f})")

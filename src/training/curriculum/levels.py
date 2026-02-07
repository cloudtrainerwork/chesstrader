"""
Progressive difficulty levels for curriculum learning in options trading.

Defines structured difficulty progression system with market complexity,
strategy sophistication, and risk parameters that gradually increase
training difficulty for stable and effective PPO training.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from ...environments.market_sim import MarketRegime
from ...strategies.base import StrategyType


class DifficultyLevel(Enum):
    """Curriculum difficulty levels with progressive complexity."""
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4


@dataclass
class CurriculumParameters:
    """
    Configuration parameters for a curriculum difficulty level.

    Attributes:
        market_volatility: (min, max) volatility range for simulation
        regime_stability: 0-1, higher = more stable regimes
        position_complexity: 1-4, strategy complexity level
        episode_length_range: (min, max) episode steps
        risk_multiplier: Risk scaling factor for position sizing
        allowed_strategies: List of strategies available at this level
        market_regimes: List of market regimes available at this level
    """
    market_volatility: Tuple[float, float]
    regime_stability: float
    position_complexity: int
    episode_length_range: Tuple[int, int]
    risk_multiplier: float
    allowed_strategies: List[StrategyType]
    market_regimes: List[MarketRegime]

    def __post_init__(self):
        """Validate parameters after initialization."""
        if not 0 <= self.regime_stability <= 1:
            raise ValueError("regime_stability must be between 0 and 1")
        if not 1 <= self.position_complexity <= 4:
            raise ValueError("position_complexity must be between 1 and 4")
        if self.market_volatility[0] >= self.market_volatility[1]:
            raise ValueError("min volatility must be less than max volatility")
        if self.episode_length_range[0] >= self.episode_length_range[1]:
            raise ValueError("min episode length must be less than max episode length")


class CurriculumLevel:
    """
    Manages configuration and advancement criteria for a specific difficulty level.

    Each level defines the market conditions, strategy complexity, and
    performance thresholds required for progression.
    """

    def __init__(self, level: DifficultyLevel):
        """
        Initialize curriculum level with predefined parameters.

        Args:
            level: The difficulty level enum value
        """
        self.level = level
        self._parameters = self._get_level_parameters()
        self._advancement_criteria = self._get_advancement_criteria()

    def _get_level_parameters(self) -> CurriculumParameters:
        """Get parameters for the current difficulty level."""

        if self.level == DifficultyLevel.BEGINNER:
            return CurriculumParameters(
                market_volatility=(0.01, 0.02),  # Low volatility
                regime_stability=0.9,  # Very stable regimes
                position_complexity=1,  # Simple strategies only
                episode_length_range=(50, 100),  # Short episodes
                risk_multiplier=0.5,  # Conservative risk
                allowed_strategies=[
                    StrategyType.LONG_CALL,
                    StrategyType.LONG_PUT,
                    StrategyType.BULL_CALL_SPREAD,
                    StrategyType.BEAR_PUT_SPREAD
                ],
                market_regimes=[
                    MarketRegime.TRENDING_UP,
                    MarketRegime.LOW_VOLATILITY
                ]
            )

        elif self.level == DifficultyLevel.INTERMEDIATE:
            return CurriculumParameters(
                market_volatility=(0.015, 0.025),  # Moderate volatility
                regime_stability=0.7,  # Moderate regime changes
                position_complexity=2,  # Vertical spreads
                episode_length_range=(75, 150),  # Medium episodes
                risk_multiplier=0.75,  # Moderate risk
                allowed_strategies=[
                    StrategyType.LONG_CALL,
                    StrategyType.LONG_PUT,
                    StrategyType.BULL_CALL_SPREAD,
                    StrategyType.BEAR_CALL_SPREAD,
                    StrategyType.BULL_PUT_SPREAD,
                    StrategyType.BEAR_PUT_SPREAD,
                    StrategyType.LONG_STRADDLE,
                    StrategyType.LONG_STRANGLE
                ],
                market_regimes=[
                    MarketRegime.TRENDING_UP,
                    MarketRegime.TRENDING_DOWN,
                    MarketRegime.MEAN_REVERTING,
                    MarketRegime.LOW_VOLATILITY
                ]
            )

        elif self.level == DifficultyLevel.ADVANCED:
            return CurriculumParameters(
                market_volatility=(0.02, 0.035),  # High volatility
                regime_stability=0.5,  # Frequent regime changes
                position_complexity=3,  # Complex strategies
                episode_length_range=(100, 200),  # Longer episodes
                risk_multiplier=1.0,  # Full risk
                allowed_strategies=[
                    StrategyType.LONG_CALL,
                    StrategyType.SHORT_CALL,
                    StrategyType.LONG_PUT,
                    StrategyType.SHORT_PUT,
                    StrategyType.BULL_CALL_SPREAD,
                    StrategyType.BEAR_CALL_SPREAD,
                    StrategyType.BULL_PUT_SPREAD,
                    StrategyType.BEAR_PUT_SPREAD,
                    StrategyType.LONG_STRADDLE,
                    StrategyType.SHORT_STRADDLE,
                    StrategyType.LONG_STRANGLE,
                    StrategyType.SHORT_STRANGLE,
                    StrategyType.IRON_CONDOR
                ],
                market_regimes=[
                    MarketRegime.TRENDING_UP,
                    MarketRegime.TRENDING_DOWN,
                    MarketRegime.MEAN_REVERTING,
                    MarketRegime.LOW_VOLATILITY,
                    MarketRegime.HIGH_VOLATILITY
                ]
            )

        elif self.level == DifficultyLevel.EXPERT:
            return CurriculumParameters(
                market_volatility=(0.025, 0.05),  # Very high volatility
                regime_stability=0.3,  # Highly unstable regimes
                position_complexity=4,  # All strategies
                episode_length_range=(150, 300),  # Long episodes
                risk_multiplier=1.25,  # Aggressive risk
                allowed_strategies=list(StrategyType),  # All strategies
                market_regimes=list(MarketRegime)  # All regimes
            )

        else:
            raise ValueError(f"Unknown difficulty level: {self.level}")

    def _get_advancement_criteria(self) -> Dict[str, float]:
        """Get performance thresholds required to advance from this level."""

        base_criteria = {
            'min_success_rate': 0.7,  # 70% success rate
            'min_sharpe_ratio': 1.0,   # Sharpe ratio >= 1.0
            'max_drawdown': 0.15,      # Max 15% drawdown
            'min_episodes': 10         # Minimum episodes at level
        }

        # Adjust criteria based on difficulty level
        if self.level == DifficultyLevel.BEGINNER:
            base_criteria['min_success_rate'] = 0.8  # Higher success rate for beginners
            base_criteria['min_sharpe_ratio'] = 0.5   # Lower Sharpe requirement

        elif self.level == DifficultyLevel.INTERMEDIATE:
            base_criteria['min_success_rate'] = 0.75
            base_criteria['min_sharpe_ratio'] = 0.75

        elif self.level == DifficultyLevel.ADVANCED:
            base_criteria['min_success_rate'] = 0.7
            base_criteria['min_sharpe_ratio'] = 1.0
            base_criteria['min_episodes'] = 15

        elif self.level == DifficultyLevel.EXPERT:
            # No advancement from expert level
            base_criteria['min_success_rate'] = float('inf')

        return base_criteria

    def get_parameters(self) -> CurriculumParameters:
        """
        Get the curriculum parameters for this level.

        Returns:
            CurriculumParameters for this difficulty level
        """
        return self._parameters

    def meets_advancement_criteria(self, performance_history: List[float],
                                 metrics_history: List[Dict[str, float]]) -> bool:
        """
        Check if performance meets criteria to advance to next level.

        Args:
            performance_history: List of recent episode returns
            metrics_history: List of dictionaries containing performance metrics

        Returns:
            True if advancement criteria are met, False otherwise
        """
        if not performance_history or not metrics_history:
            return False

        # Check minimum episodes
        if len(performance_history) < self._advancement_criteria['min_episodes']:
            return False

        # Calculate recent performance metrics
        recent_returns = performance_history[-int(self._advancement_criteria['min_episodes']):]
        recent_metrics = metrics_history[-int(self._advancement_criteria['min_episodes']):]

        # Success rate (positive returns)
        success_rate = sum(1 for r in recent_returns if r > 0) / len(recent_returns)
        if success_rate < self._advancement_criteria['min_success_rate']:
            return False

        # Sharpe ratio
        returns_array = np.array(recent_returns)
        if len(returns_array) > 1:
            sharpe_ratio = np.mean(returns_array) / (np.std(returns_array) + 1e-8)
            if sharpe_ratio < self._advancement_criteria['min_sharpe_ratio']:
                return False

        # Maximum drawdown
        cumulative_returns = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (running_max - cumulative_returns) / (running_max + 1e-8)
        max_drawdown = np.max(drawdowns)

        if max_drawdown > self._advancement_criteria['max_drawdown']:
            return False

        return True

    def meets_reduction_criteria(self, performance_history: List[float],
                               metrics_history: List[Dict[str, float]],
                               episodes_threshold: int = 20) -> bool:
        """
        Check if performance is poor enough to reduce difficulty level.

        Args:
            performance_history: List of recent episode returns
            metrics_history: List of dictionaries containing performance metrics
            episodes_threshold: Number of episodes to evaluate for reduction

        Returns:
            True if reduction criteria are met, False otherwise
        """
        if not performance_history or len(performance_history) < episodes_threshold:
            return False

        # Evaluate recent performance
        recent_returns = performance_history[-episodes_threshold:]

        # Check for persistently poor performance
        success_rate = sum(1 for r in recent_returns if r > 0) / len(recent_returns)

        # Reduce if success rate is very low
        if success_rate < 0.3:  # 30% success rate threshold
            return True

        # Check for severe drawdown
        returns_array = np.array(recent_returns)
        cumulative_returns = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (running_max - cumulative_returns) / (running_max + 1e-8)
        max_drawdown = np.max(drawdowns)

        if max_drawdown > 0.25:  # 25% drawdown threshold
            return True

        return False

    def get_next_level(self) -> Optional['CurriculumLevel']:
        """
        Get the next difficulty level.

        Returns:
            Next CurriculumLevel or None if already at maximum level
        """
        if self.level == DifficultyLevel.BEGINNER:
            return CurriculumLevel(DifficultyLevel.INTERMEDIATE)
        elif self.level == DifficultyLevel.INTERMEDIATE:
            return CurriculumLevel(DifficultyLevel.ADVANCED)
        elif self.level == DifficultyLevel.ADVANCED:
            return CurriculumLevel(DifficultyLevel.EXPERT)
        else:
            return None  # Already at maximum level

    def get_previous_level(self) -> Optional['CurriculumLevel']:
        """
        Get the previous difficulty level.

        Returns:
            Previous CurriculumLevel or None if already at minimum level
        """
        if self.level == DifficultyLevel.INTERMEDIATE:
            return CurriculumLevel(DifficultyLevel.BEGINNER)
        elif self.level == DifficultyLevel.ADVANCED:
            return CurriculumLevel(DifficultyLevel.INTERMEDIATE)
        elif self.level == DifficultyLevel.EXPERT:
            return CurriculumLevel(DifficultyLevel.ADVANCED)
        else:
            return None  # Already at minimum level

    def __str__(self) -> str:
        """String representation of curriculum level."""
        return f"CurriculumLevel({self.level.name})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"CurriculumLevel(level={self.level.name}, "
                f"complexity={self._parameters.position_complexity}, "
                f"volatility={self._parameters.market_volatility})")


def get_level_for_performance(success_rate: float, sharpe_ratio: float,
                            current_level: DifficultyLevel) -> DifficultyLevel:
    """
    Recommend appropriate difficulty level based on performance metrics.

    Args:
        success_rate: Current success rate (0-1)
        sharpe_ratio: Current Sharpe ratio
        current_level: Current difficulty level

    Returns:
        Recommended DifficultyLevel
    """
    # Performance-based level recommendation
    if success_rate >= 0.85 and sharpe_ratio >= 1.5:
        # Excellent performance - can handle expert level
        return DifficultyLevel.EXPERT
    elif success_rate >= 0.75 and sharpe_ratio >= 1.0:
        # Good performance - advanced level
        return DifficultyLevel.ADVANCED
    elif success_rate >= 0.65 and sharpe_ratio >= 0.5:
        # Moderate performance - intermediate level
        return DifficultyLevel.INTERMEDIATE
    else:
        # Poor performance - beginner level
        return DifficultyLevel.BEGINNER


def create_custom_level(market_volatility: Tuple[float, float],
                       regime_stability: float,
                       position_complexity: int,
                       episode_length_range: Tuple[int, int],
                       risk_multiplier: float,
                       allowed_strategies: List[StrategyType],
                       market_regimes: List[MarketRegime]) -> CurriculumParameters:
    """
    Create custom curriculum parameters for testing or special scenarios.

    Args:
        market_volatility: (min, max) volatility range
        regime_stability: 0-1, stability of market regimes
        position_complexity: 1-4, strategy complexity level
        episode_length_range: (min, max) episode steps
        risk_multiplier: Risk scaling factor
        allowed_strategies: List of allowed strategy types
        market_regimes: List of allowed market regimes

    Returns:
        Custom CurriculumParameters
    """
    return CurriculumParameters(
        market_volatility=market_volatility,
        regime_stability=regime_stability,
        position_complexity=position_complexity,
        episode_length_range=episode_length_range,
        risk_multiplier=risk_multiplier,
        allowed_strategies=allowed_strategies,
        market_regimes=market_regimes
    )
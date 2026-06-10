"""
Scoring engine for converting neural network outputs to actionable scores.

Implements risk-adjusted scoring, position sizing via Kelly criterion, and
multi-criteria ranking for strategy selection.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from ..strategies.base import StrategyType

logger = logging.getLogger(__name__)


@dataclass
class ScoredStrategy:
    """Container for scored strategy with all metrics."""
    strategy_type: StrategyType
    raw_probability: float  # Neural network output
    risk_adjusted_score: float  # 0-100 normalized score
    expected_value: float  # Expected P/L
    kelly_size: Optional[float]  # Position size recommendation (0-1); None if no historical data
    max_drawdown: float  # Historical/estimated max drawdown
    var_95: float  # 95% Value at Risk
    confidence: float  # Overall confidence (0-1)
    rank: int  # Final ranking among all strategies
    explanation: str  # Human-readable rationale


class ScoringEngine:
    """
    Converts neural network strategy rankings to actionable scores.

    Performs risk adjustment, position sizing, and multi-criteria ranking
    to provide interpretable strategy recommendations.
    """

    def __init__(self,
                 risk_free_rate: float = 0.04,
                 kelly_fraction: float = 0.25,  # Use fractional Kelly for safety
                 min_confidence: float = 0.3,
                 max_position_size: float = 0.2,  # Max 20% of capital per position
                 drawdown_penalty_weight: float = 2.0,
                 var_penalty_weight: float = 1.5):
        """
        Initialize scoring engine.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculations
            kelly_fraction: Fraction of Kelly criterion to use (safety)
            min_confidence: Minimum confidence threshold
            max_position_size: Maximum position size as fraction of capital
            drawdown_penalty_weight: Weight for max drawdown penalty
            var_penalty_weight: Weight for VaR penalty
        """
        self.risk_free_rate = risk_free_rate
        self.kelly_fraction = kelly_fraction
        self.min_confidence = min_confidence
        self.max_position_size = max_position_size
        self.drawdown_penalty_weight = drawdown_penalty_weight
        self.var_penalty_weight = var_penalty_weight

        # Cache for historical performance data
        self._performance_cache: Dict[StrategyType, Dict[str, float]] = {}

    def score_strategies(self,
                        strategy_probabilities: Dict[StrategyType, float],
                        risk_metrics: Dict[StrategyType, Dict[str, float]],
                        expected_returns: Dict[StrategyType, float],
                        regime_confidence: float = 1.0) -> List[ScoredStrategy]:
        """
        Score and rank strategies based on neural network outputs and risk metrics.

        Args:
            strategy_probabilities: Neural network probability outputs
            risk_metrics: Risk metrics per strategy (drawdown, VaR, etc.)
            expected_returns: Expected return per strategy
            regime_confidence: Confidence in regime detection (0-1)

        Returns:
            List of scored strategies sorted by rank
        """
        scored_strategies = []

        for strategy_type, probability in strategy_probabilities.items():
            # Get risk metrics for this strategy
            metrics = risk_metrics.get(strategy_type, {})
            expected_return = expected_returns.get(strategy_type, 0)

            # Calculate risk-adjusted score
            risk_adjusted = self._calculate_risk_adjusted_score(
                probability,
                metrics,
                regime_confidence
            )

            # Calculate expected value
            expected_value = self._calculate_expected_value(
                expected_return,
                probability,
                metrics.get('win_rate', 0.5)
            )

            # Calculate Kelly position size (None when historical data is absent)
            kelly_size = self._calculate_kelly_sizing(metrics)

            # Normalize score to 0-100 range
            normalized_score = self._normalize_score(risk_adjusted)

            # Calculate overall confidence
            confidence = self._calculate_confidence(
                probability,
                regime_confidence,
                metrics.get('backtest_samples', 0)
            )

            # Generate explanation
            explanation = self._generate_explanation(
                strategy_type,
                probability,
                normalized_score,
                metrics,
                regime_confidence
            )

            scored_strategies.append(ScoredStrategy(
                strategy_type=strategy_type,
                raw_probability=probability,
                risk_adjusted_score=normalized_score,
                expected_value=expected_value,
                kelly_size=kelly_size,
                max_drawdown=metrics.get('max_drawdown', 0.15),
                var_95=metrics.get('var_95', 0.05),
                confidence=confidence,
                rank=0,  # Will be set after sorting
                explanation=explanation
            ))

        # Sort and rank strategies
        scored_strategies = self._apply_multi_criteria_ranking(scored_strategies)

        # Apply historical performance adjustments if available
        if self._performance_cache:
            scored_strategies = self._adjust_for_historical_performance(
                scored_strategies
            )

        return scored_strategies

    def _calculate_risk_adjusted_score(self,
                                      probability: float,
                                      metrics: Dict[str, float],
                                      regime_confidence: float) -> float:
        """Calculate risk-adjusted score combining probability and risk metrics."""
        base_score = probability * regime_confidence

        # Apply risk penalties
        drawdown = metrics.get('max_drawdown', 0.15)
        var_95 = metrics.get('var_95', 0.05)

        drawdown_penalty = min(drawdown * self.drawdown_penalty_weight, 0.5)
        var_penalty = min(var_95 * self.var_penalty_weight, 0.3)

        # Calculate Sharpe ratio component
        returns = metrics.get('expected_return', 0)
        volatility = metrics.get('volatility', 0.2)

        if volatility > 0:
            sharpe = (returns - self.risk_free_rate) / volatility
            sharpe_boost = max(0, min(sharpe * 0.1, 0.2))  # Cap boost at 0.2
        else:
            sharpe_boost = 0

        # Combine components
        adjusted_score = base_score * (1 - drawdown_penalty - var_penalty) + sharpe_boost

        return max(0, min(adjusted_score, 1.0))

    def _calculate_expected_value(self,
                                 expected_return: float,
                                 probability: float,
                                 win_rate: float) -> float:
        """Calculate expected value of strategy."""
        # Simple expected value calculation
        # Can be enhanced with more sophisticated models
        return expected_return * probability * win_rate

    def _calculate_kelly_sizing(self, metrics: Dict[str, float]) -> Optional[float]:
        """
        Calculate position size using Kelly criterion.

        Kelly formula: f = (p*b - q)/b
        where:
        - f = fraction of capital to bet
        - p = probability of winning
        - q = probability of losing (1-p)
        - b = ratio of win to loss

        Returns None when the required historical inputs (win_rate, avg_win,
        avg_loss) are not all present. Sizing has no statistical basis without
        them, and returning 0 would be indistinguishable from a real "no edge"
        result (Kelly = 0). Callers must treat None as "insufficient data".
        """
        if not all(key in metrics for key in ("win_rate", "avg_win", "avg_loss")):
            return None

        win_rate = metrics["win_rate"]
        avg_win = metrics["avg_win"]
        avg_loss = metrics["avg_loss"]

        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0

        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss

        if b <= 0:
            return 0.0

        # Full Kelly
        kelly_full = (p * b - q) / b

        # Apply fractional Kelly for safety
        kelly_fraction = kelly_full * self.kelly_fraction

        # Cap at maximum position size
        return max(0.0, min(kelly_fraction, self.max_position_size))

    def _normalize_score(self, raw_score: float) -> float:
        """Normalize score to 0-100 range for interpretability."""
        # Apply sigmoid-like transformation for better distribution
        # Maps [0,1] -> [0,100] with emphasis on middle range
        normalized = 100 * (1 / (1 + np.exp(-10 * (raw_score - 0.5))))
        return round(normalized, 1)

    def _calculate_confidence(self,
                             probability: float,
                             regime_confidence: float,
                             backtest_samples: int) -> float:
        """Calculate overall confidence in recommendation."""
        # Base confidence from neural network
        base_confidence = probability

        # Adjust for regime confidence
        regime_adjusted = base_confidence * regime_confidence

        # Adjust for data quality (more samples = higher confidence)
        if backtest_samples > 0:
            data_quality = min(1.0, np.log10(backtest_samples + 1) / 3)  # Log scale
        else:
            data_quality = 0.5  # Default if no backtest data

        # Weighted average
        confidence = 0.5 * regime_adjusted + 0.3 * data_quality + 0.2 * base_confidence

        return min(1.0, confidence)

    def _generate_explanation(self,
                            strategy_type: StrategyType,
                            probability: float,
                            score: float,
                            metrics: Dict[str, float],
                            regime_confidence: float) -> str:
        """Generate human-readable explanation for score."""
        explanations = []

        # Neural network confidence
        if probability > 0.7:
            explanations.append(f"Strong neural network preference ({probability:.1%})")
        elif probability > 0.5:
            explanations.append(f"Moderate neural network preference ({probability:.1%})")
        else:
            explanations.append(f"Weak neural network preference ({probability:.1%})")

        # Regime fit
        if regime_confidence > 0.8:
            explanations.append("High regime confidence")
        elif regime_confidence < 0.5:
            explanations.append("Low regime confidence")

        # Risk metrics
        drawdown = metrics.get('max_drawdown', 0.15)
        if drawdown < 0.1:
            explanations.append(f"Low risk (max drawdown {drawdown:.1%})")
        elif drawdown > 0.2:
            explanations.append(f"Higher risk (max drawdown {drawdown:.1%})")

        # Expected return
        exp_return = metrics.get('expected_return', 0)
        if exp_return > 0.1:
            explanations.append(f"Strong expected return ({exp_return:.1%})")
        elif exp_return < 0:
            explanations.append(f"Negative expected return ({exp_return:.1%})")

        # Win rate
        win_rate = metrics.get('win_rate', 0.5)
        if win_rate > 0.6:
            explanations.append(f"High win rate ({win_rate:.1%})")
        elif win_rate < 0.4:
            explanations.append(f"Low win rate ({win_rate:.1%})")

        return f"{strategy_type.value}: " + ". ".join(explanations)

    def _apply_multi_criteria_ranking(self,
                                     strategies: List[ScoredStrategy]) -> List[ScoredStrategy]:
        """Apply multi-criteria ranking combining multiple factors."""
        # Sort by multiple criteria with weights
        # Primary: risk-adjusted score
        # Secondary: expected value
        # Tertiary: confidence

        strategies.sort(
            key=lambda s: (
                s.risk_adjusted_score * 0.5 +
                s.expected_value * 0.3 +
                s.confidence * 0.2
            ),
            reverse=True
        )

        # Assign ranks
        for i, strategy in enumerate(strategies):
            strategy.rank = i + 1

        return strategies

    def _adjust_for_historical_performance(self,
                                          strategies: List[ScoredStrategy]) -> List[ScoredStrategy]:
        """Adjust scores based on historical performance if available."""
        for strategy in strategies:
            if strategy.strategy_type in self._performance_cache:
                hist_data = self._performance_cache[strategy.strategy_type]

                # Adjust score based on historical Sharpe ratio
                hist_sharpe = hist_data.get('sharpe_ratio', 0)
                if hist_sharpe > 1.5:
                    strategy.risk_adjusted_score *= 1.1  # 10% boost
                elif hist_sharpe < 0.5:
                    strategy.risk_adjusted_score *= 0.9  # 10% penalty

                # Update explanation
                strategy.explanation += f". Historical Sharpe: {hist_sharpe:.2f}"

        # Re-rank after adjustments
        return self._apply_multi_criteria_ranking(strategies)

    def update_performance_cache(self,
                                strategy_type: StrategyType,
                                performance_data: Dict[str, float]):
        """Update historical performance cache for a strategy."""
        self._performance_cache[strategy_type] = performance_data
        logger.info(f"Updated performance cache for {strategy_type.value}")

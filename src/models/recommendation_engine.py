"""
Recommendation engine for providing actionable strategy selections.

Provides final strategy recommendations with confidence filtering, explanations,
and clean API interface.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

import numpy as np
import pandas as pd

from .scoring_engine import ScoringEngine, ScoredStrategy
from .integrated_selector import IntegratedStrategySelector
from ..strategies.base import StrategyType
from ..data.regime_labeler import RegimeType

logger = logging.getLogger(__name__)


@dataclass
class StrategyRecommendation:
    """Complete strategy recommendation with all context."""
    strategy_type: StrategyType
    score: float  # 0-100 normalized score
    confidence: float  # 0-1 confidence level
    position_size: float  # Recommended position size (fraction of capital)
    expected_return: float  # Expected return
    max_risk: float  # Maximum risk (drawdown)
    regime: RegimeType  # Current market regime
    explanation: str  # Human-readable explanation
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class RecommendationEngine:
    """
    Provides final strategy recommendations with filtering and explanations.

    Integrates neural network predictions, scoring engine, and market conditions
    to produce actionable trading recommendations.
    """

    def __init__(self,
                 integrated_selector: IntegratedStrategySelector,
                 scoring_engine: ScoringEngine,
                 confidence_threshold: float = 0.4,
                 max_recommendations: int = 3,
                 min_score: float = 40.0,
                 track_history: bool = True):
        """
        Initialize recommendation engine.

        Args:
            integrated_selector: Neural network selector with regime integration
            scoring_engine: Scoring engine for risk adjustment
            confidence_threshold: Minimum confidence for recommendations
            max_recommendations: Maximum number of recommendations to return
            min_score: Minimum score (0-100) for recommendations
            track_history: Whether to track recommendation history
        """
        self.selector = integrated_selector
        self.scoring_engine = scoring_engine
        self.confidence_threshold = confidence_threshold
        self.max_recommendations = max_recommendations
        self.min_score = min_score
        self.track_history = track_history

        # History tracking
        self.recommendation_history: List[StrategyRecommendation] = []

    def get_top_recommendations(self,
                               market_data: Dict[str, Any],
                               regime_features: np.ndarray,
                               risk_metrics: Optional[Dict[StrategyType, Dict[str, float]]] = None,
                               expected_returns: Optional[Dict[StrategyType, float]] = None) -> List[StrategyRecommendation]:
        """
        Get top strategy recommendations based on current market conditions.

        Args:
            market_data: Current market data (prices, volumes, etc.)
            regime_features: Features for regime detection
            risk_metrics: Risk metrics per strategy (if available)
            expected_returns: Expected returns per strategy (if available)

        Returns:
            List of top strategy recommendations
        """
        # Get neural network predictions and regime
        rankings, regime_type, regime_confidence = self.selector.rank_strategies_with_regime(
            market_data,
            regime_features
        )

        # Convert rankings to probability dictionary
        strategy_probabilities = {
            strategy: float(prob) for strategy, prob in rankings
        }

        # Use default risk metrics if not provided
        if risk_metrics is None:
            risk_metrics = self._get_default_risk_metrics()

        # Use default expected returns if not provided
        if expected_returns is None:
            expected_returns = self._estimate_expected_returns(
                strategy_probabilities,
                regime_type
            )

        # Score strategies
        scored_strategies = self.scoring_engine.score_strategies(
            strategy_probabilities,
            risk_metrics,
            expected_returns,
            regime_confidence
        )

        # Filter by confidence and score
        filtered = self._apply_confidence_filter(scored_strategies)

        # Validate against market conditions
        validated = self._validate_market_conditions(filtered, regime_type)

        # Create recommendations
        recommendations = []
        for strategy in validated[:self.max_recommendations]:
            recommendation = self._create_recommendation(
                strategy,
                regime_type,
                regime_confidence,
                market_data
            )
            recommendations.append(recommendation)

            # Track history
            if self.track_history:
                self.recommendation_history.append(recommendation)

        return recommendations

    def get_recommendations(self,
                            market_data: Dict[str, Any],
                            regime_features: np.ndarray,
                            risk_metrics: Optional[Dict[StrategyType, Dict[str, float]]] = None,
                            expected_returns: Optional[Dict[StrategyType, float]] = None) -> List[StrategyRecommendation]:
        """
        Compatibility wrapper for legacy callers.

        Args:
            market_data: Current market data
            regime_features: Regime features
            risk_metrics: Optional risk metrics per strategy
            expected_returns: Optional expected returns per strategy

        Returns:
            List of strategy recommendations
        """
        return self.get_top_recommendations(
            market_data,
            regime_features,
            risk_metrics=risk_metrics,
            expected_returns=expected_returns
        )

    def batch_recommend(self,
                       symbols: List[str],
                       market_data_batch: List[Dict[str, Any]],
                       regime_features_batch: List[np.ndarray]) -> Dict[str, List[StrategyRecommendation]]:
        """
        Get recommendations for multiple symbols.

        Args:
            symbols: List of symbols
            market_data_batch: Market data for each symbol
            regime_features_batch: Regime features for each symbol

        Returns:
            Dictionary mapping symbols to recommendations
        """
        batch_recommendations = {}

        for symbol, market_data, regime_features in zip(
            symbols, market_data_batch, regime_features_batch
        ):
            recommendations = self.get_top_recommendations(
                market_data,
                regime_features
            )
            batch_recommendations[symbol] = recommendations

        return batch_recommendations

    def _apply_confidence_filter(self,
                                strategies: List[ScoredStrategy]) -> List[ScoredStrategy]:
        """Apply confidence threshold filtering."""
        filtered = [
            s for s in strategies
            if s.confidence >= self.confidence_threshold
            and s.risk_adjusted_score >= self.min_score
        ]

        if not filtered:
            # If nothing passes filter, return top strategy with warning
            logger.warning(
                f"No strategies meet confidence threshold {self.confidence_threshold}. "
                f"Returning top strategy with reduced confidence."
            )
            if strategies:
                return [strategies[0]]  # Return best available
            return []

        return filtered

    def _validate_market_conditions(self,
                                   strategies: List[ScoredStrategy],
                                   regime: RegimeType) -> List[ScoredStrategy]:
        """Validate recommendations against current market regime."""
        validated = []

        for strategy in strategies:
            # Check if strategy is appropriate for regime
            if self._is_strategy_valid_for_regime(strategy.strategy_type, regime):
                validated.append(strategy)
            else:
                logger.debug(
                    f"Strategy {strategy.strategy_type.value} not optimal for {regime.value} regime"
                )

        if not validated and strategies:
            # Fallback to top strategy if none match regime constraints
            validated = [strategies[0]]

        return validated

    def _is_strategy_valid_for_regime(self,
                                     strategy: StrategyType,
                                     regime: RegimeType) -> bool:
        """Check if strategy is appropriate for market regime."""
        # Define strategy-regime compatibility
        compatibility = {
            RegimeType.BULL_TRENDING: [
                StrategyType.BULL_CALL_SPREAD,
                StrategyType.BULL_PUT_SPREAD,
                StrategyType.SHORT_CALL,
                StrategyType.CALENDAR_CALL
            ],
            RegimeType.BEAR_TRENDING: [
                StrategyType.BEAR_PUT_SPREAD,
                StrategyType.BEAR_CALL_SPREAD,
                StrategyType.LONG_PUT,
                StrategyType.CALENDAR_PUT
            ],
            RegimeType.SIDEWAYS_RANGING: [
                StrategyType.IRON_CONDOR,
                StrategyType.BUTTERFLY,
                StrategyType.CALENDAR_CALL,
                StrategyType.CALENDAR_PUT,
                StrategyType.SHORT_STRADDLE,
                StrategyType.SHORT_STRANGLE
            ],
            RegimeType.HIGH_VOLATILITY: [
                StrategyType.LONG_STRADDLE,
                StrategyType.LONG_STRANGLE,
                StrategyType.CALENDAR_CALL,
                StrategyType.CALENDAR_PUT
            ],
            RegimeType.LOW_VOLATILITY: [
                StrategyType.IRON_CONDOR,
                StrategyType.BUTTERFLY,
                StrategyType.SHORT_STRADDLE,
                StrategyType.SHORT_STRANGLE,
                StrategyType.SHORT_CALL
            ]
        }

        # Get compatible strategies for regime
        compatible_strategies = compatibility.get(regime, [])

        # Allow strategy if it's in the compatible list
        # or if it's not strongly contraindicated
        return strategy in compatible_strategies or self._is_neutral_strategy(strategy)

    def _is_neutral_strategy(self, strategy: StrategyType) -> bool:
        """Check if strategy is regime-neutral."""
        neutral_strategies = [
            StrategyType.IRON_CONDOR,
            StrategyType.BUTTERFLY,
            StrategyType.CALENDAR_CALL,
            StrategyType.CALENDAR_PUT
        ]
        return strategy in neutral_strategies

    def _create_recommendation(self,
                              scored_strategy: ScoredStrategy,
                              regime: RegimeType,
                              regime_confidence: float,
                              market_data: Dict[str, Any]) -> StrategyRecommendation:
        """Create complete recommendation from scored strategy."""
        # Generate detailed explanation
        explanation = self._generate_recommendation_explanation(
            scored_strategy,
            regime,
            regime_confidence,
            market_data
        )

        # Create metadata
        metadata = {
            'regime_confidence': regime_confidence,
            'neural_network_probability': scored_strategy.raw_probability,
            'rank': scored_strategy.rank,
            'var_95': scored_strategy.var_95,
            'market_data_snapshot': {
                'price': market_data.get('current_price', 0),
                'volume': market_data.get('volume', 0),
                'volatility': market_data.get('implied_volatility', 0)
            }
        }

        return StrategyRecommendation(
            strategy_type=scored_strategy.strategy_type,
            score=scored_strategy.risk_adjusted_score,
            confidence=scored_strategy.confidence,
            position_size=scored_strategy.kelly_size,
            expected_return=scored_strategy.expected_value,
            max_risk=scored_strategy.max_drawdown,
            regime=regime,
            explanation=explanation,
            metadata=metadata
        )

    def _generate_recommendation_explanation(self,
                                           strategy: ScoredStrategy,
                                           regime: RegimeType,
                                           regime_confidence: float,
                                           market_data: Dict[str, Any]) -> str:
        """Generate detailed explanation for recommendation."""
        explanations = []

        # Strategy and score
        explanations.append(
            f"**{strategy.strategy_type.value}** (Score: {strategy.risk_adjusted_score:.1f}/100)"
        )

        # Market regime context
        explanations.append(
            f"Market regime: {regime.value} (confidence: {regime_confidence:.1%})"
        )

        # Risk-return profile
        explanations.append(
            f"Expected return: {strategy.expected_value:.2%}, "
            f"Max risk: {strategy.max_drawdown:.1%}"
        )

        # Position sizing (kelly_size is None when historical data is absent)
        if strategy.kelly_size is None:
            explanations.append(
                "Recommended position size: insufficient historical data to size"
            )
        else:
            explanations.append(
                f"Recommended position size: {strategy.kelly_size:.1%} of capital"
            )

        # Add strategy-specific insights
        if strategy.strategy_type in [StrategyType.IRON_CONDOR, StrategyType.BUTTERFLY]:
            explanations.append(
                "Neutral strategy benefiting from time decay and range-bound markets"
            )
        elif strategy.strategy_type in [StrategyType.BULL_CALL_SPREAD, StrategyType.BULL_PUT_SPREAD]:
            explanations.append(
                "Bullish strategy with defined risk, suitable for upward trend"
            )
        elif strategy.strategy_type in [StrategyType.LONG_STRADDLE, StrategyType.LONG_STRANGLE]:
            explanations.append(
                "Volatility play expecting significant price movement"
            )

        # Market conditions
        if 'implied_volatility' in market_data:
            iv = market_data['implied_volatility']
            if iv > 0.3:
                explanations.append(f"High IV ({iv:.1%}) favors premium selling")
            elif iv < 0.15:
                explanations.append(f"Low IV ({iv:.1%}) favors premium buying")

        # Add base explanation from scoring
        explanations.append(strategy.explanation)

        return " | ".join(explanations)

    def _get_default_risk_metrics(self) -> Dict[StrategyType, Dict[str, float]]:
        """Get default risk metrics for strategies."""
        # Default conservative estimates
        defaults = {
            StrategyType.IRON_CONDOR: {
                'max_drawdown': 0.10, 'var_95': 0.04, 'win_rate': 0.65,
                'avg_win': 1.0, 'avg_loss': 1.5, 'volatility': 0.15
            },
            StrategyType.BUTTERFLY: {
                'max_drawdown': 0.12, 'var_95': 0.05, 'win_rate': 0.60,
                'avg_win': 1.2, 'avg_loss': 1.5, 'volatility': 0.18
            },
            StrategyType.BULL_CALL_SPREAD: {
                'max_drawdown': 0.15, 'var_95': 0.06, 'win_rate': 0.55,
                'avg_win': 2.0, 'avg_loss': 1.0, 'volatility': 0.25
            },
            StrategyType.BEAR_PUT_SPREAD: {
                'max_drawdown': 0.15, 'var_95': 0.06, 'win_rate': 0.55,
                'avg_win': 2.0, 'avg_loss': 1.0, 'volatility': 0.25
            },
            StrategyType.LONG_STRADDLE: {
                'max_drawdown': 0.25, 'var_95': 0.10, 'win_rate': 0.40,
                'avg_win': 3.0, 'avg_loss': 1.0, 'volatility': 0.35
            },
            StrategyType.SHORT_STRADDLE: {
                'max_drawdown': 0.30, 'var_95': 0.12, 'win_rate': 0.60,
                'avg_win': 1.0, 'avg_loss': 3.0, 'volatility': 0.30
            },
            StrategyType.LONG_STRANGLE: {
                'max_drawdown': 0.20, 'var_95': 0.08, 'win_rate': 0.35,
                'avg_win': 4.0, 'avg_loss': 1.0, 'volatility': 0.30
            },
            StrategyType.SHORT_STRANGLE: {
                'max_drawdown': 0.25, 'var_95': 0.10, 'win_rate': 0.65,
                'avg_win': 1.0, 'avg_loss': 2.5, 'volatility': 0.25
            },
            StrategyType.CALENDAR_CALL: {
                'max_drawdown': 0.08, 'var_95': 0.03, 'win_rate': 0.60,
                'avg_win': 1.5, 'avg_loss': 1.0, 'volatility': 0.12
            },
            StrategyType.CALENDAR_PUT: {
                'max_drawdown': 0.08, 'var_95': 0.03, 'win_rate': 0.60,
                'avg_win': 1.5, 'avg_loss': 1.0, 'volatility': 0.12
            },
            StrategyType.BULL_PUT_SPREAD: {
                'max_drawdown': 0.15, 'var_95': 0.06, 'win_rate': 0.60,
                'avg_win': 1.0, 'avg_loss': 2.0, 'volatility': 0.22
            },
            StrategyType.BEAR_CALL_SPREAD: {
                'max_drawdown': 0.15, 'var_95': 0.06, 'win_rate': 0.60,
                'avg_win': 1.0, 'avg_loss': 2.0, 'volatility': 0.22
            }
        }

        # Add default values for any missing strategies
        for strategy in StrategyType:
            if strategy not in defaults:
                defaults[strategy] = {
                    'max_drawdown': 0.15, 'var_95': 0.06, 'win_rate': 0.50,
                    'avg_win': 1.5, 'avg_loss': 1.5, 'volatility': 0.20
                }

        return defaults

    def _estimate_expected_returns(self,
                                  probabilities: Dict[StrategyType, float],
                                  regime: RegimeType) -> Dict[StrategyType, float]:
        """Estimate expected returns based on probabilities and regime."""
        # Base returns by strategy type
        base_returns = {
            StrategyType.IRON_CONDOR: 0.08,
            StrategyType.BUTTERFLY: 0.10,
            StrategyType.BULL_CALL_SPREAD: 0.15,
            StrategyType.BEAR_PUT_SPREAD: 0.15,
            StrategyType.LONG_STRADDLE: 0.20,
            StrategyType.SHORT_STRADDLE: 0.12,
            StrategyType.LONG_STRANGLE: 0.25,
            StrategyType.SHORT_STRANGLE: 0.10,
            StrategyType.CALENDAR_CALL: 0.06,
            StrategyType.CALENDAR_PUT: 0.06,
            StrategyType.SHORT_CALL: 0.05,
            StrategyType.LONG_PUT: 0.03,
            StrategyType.BULL_PUT_SPREAD: 0.12,
            StrategyType.BEAR_CALL_SPREAD: 0.12
        }

        # Adjust for regime
        regime_multipliers = {
            RegimeType.BULL_TRENDING: {
                StrategyType.BULL_CALL_SPREAD: 1.3,
                StrategyType.BULL_PUT_SPREAD: 1.2,
                StrategyType.BEAR_PUT_SPREAD: 0.7,
                StrategyType.BEAR_CALL_SPREAD: 0.7
            },
            RegimeType.BEAR_TRENDING: {
                StrategyType.BEAR_PUT_SPREAD: 1.3,
                StrategyType.BEAR_CALL_SPREAD: 1.2,
                StrategyType.BULL_CALL_SPREAD: 0.7,
                StrategyType.BULL_PUT_SPREAD: 0.7
            },
            RegimeType.HIGH_VOLATILITY: {
                StrategyType.LONG_STRADDLE: 1.4,
                StrategyType.LONG_STRANGLE: 1.4,
                StrategyType.SHORT_STRADDLE: 0.6,
                StrategyType.SHORT_STRANGLE: 0.6
            },
            RegimeType.LOW_VOLATILITY: {
                StrategyType.IRON_CONDOR: 1.2,
                StrategyType.BUTTERFLY: 1.2,
                StrategyType.SHORT_STRADDLE: 1.3,
                StrategyType.SHORT_STRANGLE: 1.3
            },
            RegimeType.SIDEWAYS_RANGING: {
                StrategyType.IRON_CONDOR: 1.3,
                StrategyType.BUTTERFLY: 1.3,
                StrategyType.CALENDAR_CALL: 1.2,
                StrategyType.CALENDAR_PUT: 1.2
            }
        }

        expected_returns = {}
        regime_mults = regime_multipliers.get(regime, {})

        for strategy in StrategyType:
            base = base_returns.get(strategy, 0.10)
            mult = regime_mults.get(strategy, 1.0)
            prob_adj = probabilities.get(strategy, 0.5)

            # Adjust return based on regime and probability
            expected_returns[strategy] = base * mult * (0.5 + prob_adj)

        return expected_returns

    def get_recommendation_history(self,
                                  limit: Optional[int] = None) -> List[StrategyRecommendation]:
        """Get recommendation history."""
        if limit:
            return self.recommendation_history[-limit:]
        return self.recommendation_history

    def clear_history(self):
        """Clear recommendation history."""
        self.recommendation_history = []
        logger.info("Recommendation history cleared")

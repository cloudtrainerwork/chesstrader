"""
Strategy factory for dynamic strategy selection and recommendations.

Provides centralized strategy instantiation and regime-based recommendation engine
for all 16 core options strategies in the framework.
"""

from typing import Dict, List, Tuple, Type, Optional
from dataclasses import dataclass

from .base import BaseStrategy, StrategyType, MarketConditions
from .neutral import IronCondorStrategy, IronButterflyStrategy
from .directional import (
    BullCallSpreadStrategy, BearCallSpreadStrategy,
    BullPutSpreadStrategy, BearPutSpreadStrategy
)
from .volatility import (
    LongStraddleStrategy, ShortStraddleStrategy,
    LongStrangleStrategy, ShortStrangleStrategy
)
from .advanced import CalendarCallStrategy, CalendarPutStrategy
from .equity import CoveredCallStrategy, CollarStrategy


@dataclass
class StrategyRecommendation:
    """
    Strategy recommendation with confidence scoring.

    Attributes:
        strategy_type: Type of strategy recommended
        strategy_instance: Instantiated strategy object
        confidence: Confidence score (0.0 to 1.0)
        reasons: List of reasons for recommendation
        expected_performance: Expected performance category
        risk_assessment: Risk level assessment for current conditions
    """
    strategy_type: StrategyType
    strategy_instance: BaseStrategy
    confidence: float
    reasons: List[str]
    expected_performance: str
    risk_assessment: str


class SingleLegStrategy(BaseStrategy):
    """
    Placeholder implementation for single leg strategies.

    These would be implemented for individual calls/puts used in
    covered call and collar strategies.
    """

    def __init__(self, option_type: str, position: str):
        self.option_type = option_type  # "call" or "put"
        self.position = position        # "long" or "short"
        super().__init__()

    def _create_metadata(self):
        from .base import StrategyMetadata, StrategyCategory, RiskLevel
        return StrategyMetadata(
            name=f"{self.position.title()} {self.option_type.title()}",
            category=StrategyCategory.ADVANCED,
            risk_level=RiskLevel.HIGH if self.position == "short" else RiskLevel.MEDIUM,
            capital_requirement=1.0 if self.position == "long" else 3.0,
            description=f"{self.position.title()} {self.option_type} option position",
            typical_market_conditions=["Various market conditions"]
        )

    def get_strategy_type(self) -> StrategyType:
        type_map = {
            ("long", "call"): StrategyType.LONG_CALL,
            ("short", "call"): StrategyType.SHORT_CALL,
            ("long", "put"): StrategyType.LONG_PUT,
            ("short", "put"): StrategyType.SHORT_PUT,
        }
        return type_map.get((self.position, self.option_type), StrategyType.LONG_CALL)

    def validate_market_conditions(self, conditions):
        return True  # Simplified for placeholder

    def calculate_entry_criteria(self, conditions):
        from .base import EntrySignal
        return EntrySignal(should_enter=False, confidence=0.0, reasons=["Placeholder implementation"])

    def calculate_exit_criteria(self, position, conditions):
        from .base import ExitSignal
        return ExitSignal(should_exit=False, urgency=0.0, exit_type="hold", reasons=["Placeholder"])

    def get_risk_metrics(self, strikes, underlying_price, time_to_expiration, volatility):
        from .base import RiskMetrics
        return RiskMetrics(
            max_profit=1000, max_loss=1000, breakeven_points=[underlying_price],
            profit_probability=0.5, risk_reward_ratio=1.0,
            capital_requirement=1000, margin_requirement=500
        )

    def get_position_legs(self, strikes, expiration_date):
        from .base import PositionLeg, OptionType
        return [PositionLeg(
            option_type=OptionType.CALL if self.option_type == "call" else OptionType.PUT,
            strike=strikes[0] if strikes else 5000,
            quantity=1 if self.position == "long" else -1,
            expiration_date=expiration_date
        )]

    def _validate_strategy_strikes(self, strikes, underlying_price):
        return len(strikes) == 1


class StrategyFactory:
    """
    Factory for creating strategy instances and providing regime-based recommendations.

    Manages all 16 core strategy types and provides intelligent strategy selection
    based on market regime detection and current conditions.
    """

    def __init__(self):
        """Initialize strategy factory with all strategy mappings."""
        self._strategy_registry = self._build_strategy_registry()
        self._regime_mappings = self._build_regime_mappings()

    def _build_strategy_registry(self) -> Dict[StrategyType, Type[BaseStrategy]]:
        """
        Build registry mapping strategy types to implementation classes.

        Returns:
            Dictionary mapping StrategyType enums to strategy classes
        """
        return {
            # Single leg strategies (using placeholder for now)
            StrategyType.LONG_CALL: lambda: SingleLegStrategy("call", "long"),
            StrategyType.SHORT_CALL: lambda: SingleLegStrategy("call", "short"),
            StrategyType.LONG_PUT: lambda: SingleLegStrategy("put", "long"),
            StrategyType.SHORT_PUT: lambda: SingleLegStrategy("put", "short"),

            # Vertical spreads
            StrategyType.BULL_CALL_SPREAD: BullCallSpreadStrategy,
            StrategyType.BEAR_CALL_SPREAD: BearCallSpreadStrategy,
            StrategyType.BULL_PUT_SPREAD: BullPutSpreadStrategy,
            StrategyType.BEAR_PUT_SPREAD: BearPutSpreadStrategy,

            # Horizontal spreads
            StrategyType.CALENDAR_CALL: CalendarCallStrategy,
            StrategyType.CALENDAR_PUT: CalendarPutStrategy,

            # Volatility strategies
            StrategyType.LONG_STRADDLE: LongStraddleStrategy,
            StrategyType.SHORT_STRADDLE: ShortStraddleStrategy,
            StrategyType.LONG_STRANGLE: LongStrangleStrategy,
            StrategyType.SHORT_STRANGLE: ShortStrangleStrategy,

            # Complex strategies
            StrategyType.IRON_CONDOR: IronCondorStrategy,
            StrategyType.BUTTERFLY: IronButterflyStrategy,
        }

    def _build_regime_mappings(self) -> Dict[int, List[StrategyType]]:
        """
        Build mappings from market regimes to preferred strategies.

        Regime integers match RegimeType in src/data/regime_labeler.py, which
        produces the training labels the regime detector learns. The mapping
        below must stay aligned with that enum so that the strategy recommended
        at inference matches the regime the model was trained to identify.

        8-regime market classification system (RegimeType):
        0: BULL_TRENDING    - Strong upward momentum
        1: BEAR_TRENDING    - Strong downward momentum
        2: HIGH_VOLATILITY  - Elevated volatility environment
        3: LOW_VOLATILITY   - Subdued volatility environment
        4: SIDEWAYS_RANGING - Consolidation patterns
        5: RECOVERY         - Post-decline bounce patterns
        6: DISTRIBUTION     - Pre-decline weakening
        7: CRISIS           - Extreme stress conditions

        Returns:
            Dictionary mapping regime numbers to preferred strategy types
        """
        return {
            0: [  # BULL_TRENDING
                StrategyType.LONG_CALL,
                StrategyType.BULL_CALL_SPREAD,
                StrategyType.BULL_PUT_SPREAD,
            ],
            1: [  # BEAR_TRENDING
                StrategyType.LONG_PUT,
                StrategyType.BEAR_PUT_SPREAD,
                StrategyType.BEAR_CALL_SPREAD,
            ],
            2: [  # HIGH_VOLATILITY
                StrategyType.LONG_STRADDLE,
                StrategyType.LONG_STRANGLE,
            ],
            3: [  # LOW_VOLATILITY
                StrategyType.CALENDAR_CALL,
                StrategyType.CALENDAR_PUT,
                StrategyType.SHORT_STRADDLE,
            ],
            4: [  # SIDEWAYS_RANGING
                StrategyType.IRON_CONDOR,
                StrategyType.SHORT_STRANGLE,
                StrategyType.BUTTERFLY,
            ],
            5: [  # RECOVERY
                StrategyType.BULL_CALL_SPREAD,
                StrategyType.LONG_CALL,
                StrategyType.BULL_PUT_SPREAD,
            ],
            6: [  # DISTRIBUTION
                StrategyType.BEAR_CALL_SPREAD,
                StrategyType.IRON_CONDOR,
                StrategyType.SHORT_STRANGLE,
            ],
            7: [  # CRISIS
                StrategyType.LONG_PUT,
                StrategyType.LONG_STRADDLE,
                StrategyType.BEAR_PUT_SPREAD,
            ],
        }

    def create_strategy(self, strategy_type: StrategyType) -> BaseStrategy:
        """
        Create strategy instance by type.

        Args:
            strategy_type: Type of strategy to create

        Returns:
            Instantiated strategy object

        Raises:
            ValueError: If strategy type is not supported
        """
        if strategy_type not in self._strategy_registry:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")

        strategy_class = self._strategy_registry[strategy_type]
        return strategy_class()

    def get_all_strategies(self) -> List[BaseStrategy]:
        """
        Get instances of all supported strategies.

        Returns:
            List of all strategy instances
        """
        strategies = []
        for strategy_type in self._strategy_registry:
            try:
                strategies.append(self.create_strategy(strategy_type))
            except Exception as e:
                # Log error but continue with other strategies
                print(f"Warning: Could not create {strategy_type}: {e}")
                continue
        return strategies

    def get_recommended_strategies(self, regime: int, conditions: Optional[MarketConditions] = None,
                                  max_recommendations: int = 5) -> List[StrategyRecommendation]:
        """
        Get ranked strategy recommendations for given market regime.

        Args:
            regime: Market regime (0-7)
            conditions: Optional market conditions for detailed analysis
            max_recommendations: Maximum number of recommendations to return

        Returns:
            List of ranked strategy recommendations with confidence scores
        """
        if regime not in self._regime_mappings:
            raise ValueError(f"Invalid market regime: {regime}. Must be 0-7")

        preferred_strategies = self._regime_mappings[regime]
        recommendations = []

        for i, strategy_type in enumerate(preferred_strategies[:max_recommendations]):
            try:
                strategy_instance = self.create_strategy(strategy_type)

                # Calculate confidence based on regime preference and market conditions
                base_confidence = 1.0 - (i * 0.15)  # Decreasing confidence by rank

                # Adjust confidence based on market conditions if provided
                if conditions:
                    if strategy_instance.validate_market_conditions(conditions):
                        confidence_boost = 0.2
                        entry_signal = strategy_instance.calculate_entry_criteria(conditions)
                        confidence_boost += entry_signal.confidence * 0.3
                    else:
                        confidence_boost = -0.3

                    final_confidence = min(1.0, max(0.0, base_confidence + confidence_boost))
                else:
                    final_confidence = base_confidence

                # Generate reasons for recommendation
                reasons = [
                    f"Strategy ranked #{i+1} for regime {regime}",
                    f"Risk level: {strategy_instance.risk_level.value}",
                ]

                if conditions and strategy_instance.validate_market_conditions(conditions):
                    reasons.append("Market conditions validate strategy suitability")

                # Assess expected performance and risk
                expected_performance = self._assess_expected_performance(strategy_instance, regime)
                risk_assessment = self._assess_risk_for_regime(strategy_instance, regime)

                recommendation = StrategyRecommendation(
                    strategy_type=strategy_type,
                    strategy_instance=strategy_instance,
                    confidence=final_confidence,
                    reasons=reasons,
                    expected_performance=expected_performance,
                    risk_assessment=risk_assessment
                )

                recommendations.append(recommendation)

            except Exception as e:
                # Skip problematic strategies but continue
                print(f"Warning: Could not create recommendation for {strategy_type}: {e}")
                continue

        # Sort by confidence descending
        recommendations.sort(key=lambda x: x.confidence, reverse=True)

        return recommendations

    def _assess_expected_performance(self, strategy: BaseStrategy, regime: int) -> str:
        """
        Assess expected performance category for strategy in given regime.

        Args:
            strategy: Strategy instance
            regime: Market regime

        Returns:
            Performance category string
        """
        regime_performance_map = {
            0: "growth",         # BULL_TRENDING
            1: "defensive",      # BEAR_TRENDING
            2: "volatile",       # HIGH_VOLATILITY
            3: "income",         # LOW_VOLATILITY
            4: "neutral",        # SIDEWAYS_RANGING
            5: "recovery",       # RECOVERY
            6: "defensive",      # DISTRIBUTION
            7: "defensive",      # CRISIS
        }

        return regime_performance_map.get(regime, "unknown")

    def _assess_risk_for_regime(self, strategy: BaseStrategy, regime: int) -> str:
        """
        Assess risk level appropriateness for strategy in given regime.

        Args:
            strategy: Strategy instance
            regime: Market regime

        Returns:
            Risk assessment string
        """
        base_risk = strategy.risk_level.value

        high_risk_regimes = [1, 2, 6, 7]  # BEAR_TRENDING, HIGH_VOLATILITY, DISTRIBUTION, CRISIS
        if regime in high_risk_regimes:
            if base_risk in ["low", "medium"]:
                return "appropriate_risk"
            else:
                return "elevated_risk"
        else:
            if base_risk in ["medium", "high"]:
                return "manageable_risk"
            else:
                return "conservative_risk"

    def get_strategy_count(self) -> int:
        """
        Get total number of supported strategies.

        Returns:
            Number of strategies in registry
        """
        return len(self._strategy_registry)

    def get_supported_strategy_types(self) -> List[StrategyType]:
        """
        Get list of all supported strategy types.

        Returns:
            List of StrategyType enums
        """
        return list(self._strategy_registry.keys())
"""
Advanced options strategies implementation.

Implements Calendar, Diagonal, Covered Call, and Collar strategies for complex
market conditions requiring multi-expiration or equity integration.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, TYPE_CHECKING

from .base import (
    BaseStrategy, StrategyMetadata, StrategyCategory, RiskLevel,
    MarketConditions, EntrySignal, ExitSignal, RiskMetrics, PositionLeg,
    OptionType, StrategyType
)

if TYPE_CHECKING:
    from ..features.position_models import Position


class CalendarCallStrategy(BaseStrategy):
    """
    Calendar Call Spread options strategy implementation.

    Calendar Call Spread consists of:
    - Short call at near expiration (sold for higher time decay)
    - Long call at far expiration with same strike (purchased for time decay protection)

    Structure: Sell Call @ A (near), Buy Call @ A (far)

    Time decay strategy profiting from theta differential between expirations.
    Net debit strategy - pay premium upfront.

    Max profit: When underlying is at strike price at short expiration
    Max loss: Net debit paid (when underlying moves significantly away from strike)
    Breakeven: Strike +/- net debit depending on direction

    Best market conditions:
    - Low volatility regimes (4) for time decay advantage
    - Neutral to slightly bullish bias
    - Avoid high volatility regimes where underlying may move significantly
    """

    def _create_metadata(self) -> StrategyMetadata:
        """Create Calendar Call Spread strategy metadata."""
        return StrategyMetadata(
            name="Calendar Call Spread",
            category=StrategyCategory.ADVANCED,
            risk_level=RiskLevel.MEDIUM,
            capital_requirement=1.2,  # Moderate risk due to time decay complexity
            description="Time decay strategy with short near-term and long far-term calls at same strike",
            typical_market_conditions=[
                "Low volatility market (regime 4)",
                "Neutral to slightly bullish bias",
                "Time decay favorable environment",
                "Stable underlying price expected near strike"
            ]
        )

    def get_strategy_type(self) -> StrategyType:
        """Get strategy type enum."""
        return StrategyType.CALENDAR_CALL

    def validate_market_conditions(self, conditions: MarketConditions) -> bool:
        """
        Validate market conditions for Calendar Call strategy.

        Args:
            conditions: Current market conditions

        Returns:
            True if conditions are suitable for Calendar Call
        """
        # Favorable regime: Low volatility (4)
        if conditions.regime != 4:
            return False

        # Check volatility rank (low volatility preferred)
        if conditions.volatility_rank > 0.5:  # Avoid high volatility
            return False

        # Check trend strength (prefer neutral to slightly bullish)
        if abs(conditions.trend_strength) > 0.6:  # Avoid strong trends
            return False

        # Check time to expiration for near leg (15-45 DTE optimal)
        if conditions.time_to_expiration < 10 or conditions.time_to_expiration > 60:
            return False

        return True

    def calculate_entry_criteria(self, conditions: MarketConditions) -> EntrySignal:
        """
        Calculate entry signal for Calendar Call Spread.

        Args:
            conditions: Current market conditions

        Returns:
            EntrySignal with entry decision and confidence
        """
        reasons = []
        confidence = 0.0

        # Base validation
        if not self.validate_market_conditions(conditions):
            return EntrySignal(
                should_enter=False,
                confidence=0.0,
                reasons=["Market conditions not suitable for calendar spread"]
            )

        # Evaluate entry factors
        # 1. Volatility assessment
        if conditions.volatility_rank < 0.3:
            confidence += 0.3
            reasons.append("Low volatility favors time decay strategy")

        # 2. Trend neutrality
        if abs(conditions.trend_strength) < 0.3:
            confidence += 0.3
            reasons.append("Neutral trend suitable for calendar spread")

        # 3. Time decay window
        if 20 <= conditions.time_to_expiration <= 40:
            confidence += 0.4
            reasons.append("Optimal time to expiration for near leg")

        # Entry decision
        should_enter = confidence >= 0.6

        return EntrySignal(
            should_enter=should_enter,
            confidence=confidence,
            reasons=reasons,
            recommended_size=1.0 if should_enter else 0.0
        )

    def calculate_exit_criteria(self, position: 'Position', conditions: MarketConditions) -> ExitSignal:
        """
        Calculate exit signal for existing Calendar Call position.

        Args:
            position: Current position
            conditions: Current market conditions

        Returns:
            ExitSignal with exit decision and reasoning
        """
        reasons = []
        urgency = 0.0

        # Check for volatility increase
        if conditions.volatility_rank > 0.7:
            urgency += 0.4
            reasons.append("High volatility threatens time decay strategy")

        # Check for strong trend emergence
        if abs(conditions.trend_strength) > 0.7:
            urgency += 0.4
            reasons.append("Strong trend may move underlying away from strike")

        # Check time decay (close when near expiration approaches)
        if conditions.time_to_expiration < 7:
            urgency += 0.5
            reasons.append("Near expiration approaching, time to close")

        # Determine exit decision
        should_exit = urgency >= 0.6
        exit_type = "defensive" if urgency > 0.6 else "time"

        return ExitSignal(
            should_exit=should_exit,
            urgency=urgency,
            exit_type=exit_type,
            reasons=reasons
        )

    def get_risk_metrics(self, strikes: List[int], underlying_price: int,
                        time_to_expiration: int, volatility: float) -> RiskMetrics:
        """
        Calculate risk metrics for Calendar Call Spread.

        Args:
            strikes: Strike prices [strike] (same for both legs)
            underlying_price: Current underlying price in cents
            time_to_expiration: Days to expiration for near leg
            volatility: Implied volatility estimate

        Returns:
            RiskMetrics with comprehensive risk analysis
        """
        if len(strikes) != 1:
            raise ValueError("Calendar spread requires exactly one strike price")

        strike = strikes[0]

        # Estimate near and far leg premiums
        # Near leg (shorter expiration, higher time decay)
        near_premium = max(underlying_price - strike, 0) + int(volatility * strike * 0.1)

        # Far leg (longer expiration, ~2x time)
        far_premium = max(underlying_price - strike, 0) + int(volatility * strike * 0.15)

        # Net debit (cost to enter)
        net_debit = far_premium - near_premium

        # Calculate risk metrics
        # Max profit occurs when stock is at strike at near expiration
        max_profit = int(strike * 0.03)  # Estimate ~3% of strike value

        # Max loss is the net debit paid
        max_loss = net_debit

        # Breakeven points (approximate)
        breakeven_lower = strike - net_debit
        breakeven_upper = strike + net_debit

        # Profit probability (higher when near strike)
        distance_from_strike = abs(underlying_price - strike) / strike
        profit_probability = max(0.3, 0.8 - distance_from_strike * 2)

        # Risk-reward ratio
        risk_reward_ratio = max_profit / max_loss if max_loss > 0 else 0.0

        # Capital requirement is the net debit
        capital_requirement = net_debit

        # Margin requirement (for spread)
        margin_requirement = max(net_debit, int(strike * 0.1))  # 10% of strike minimum

        return RiskMetrics(
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[breakeven_lower, breakeven_upper],
            profit_probability=profit_probability,
            risk_reward_ratio=risk_reward_ratio,
            capital_requirement=capital_requirement,
            margin_requirement=margin_requirement
        )

    def get_position_legs(self, strikes: List[int], expiration_date: datetime) -> List[PositionLeg]:
        """
        Get position legs for Calendar Call Spread construction.

        Args:
            strikes: Strike prices [strike] (same for both legs)
            expiration_date: Near expiration date (far will be calculated)

        Returns:
            List of PositionLeg objects defining the complete position
        """
        if len(strikes) != 1:
            raise ValueError("Calendar spread requires exactly one strike price")

        strike = strikes[0]

        # Calculate far expiration (typically 30-60 days beyond near)
        far_expiration = expiration_date + timedelta(days=35)

        # Create position legs
        legs = [
            # Short near-term call (higher time decay)
            PositionLeg(
                option_type=OptionType.CALL,
                strike=strike,
                quantity=-1,  # Short
                expiration_date=expiration_date
            ),
            # Long far-term call (protection)
            PositionLeg(
                option_type=OptionType.CALL,
                strike=strike,
                quantity=1,   # Long
                expiration_date=far_expiration
            )
        ]

        return legs

    def _validate_strategy_strikes(self, strikes: List[int], underlying_price: int) -> bool:
        """
        Validate strike configuration for Calendar Call Spread.

        Args:
            strikes: Strike prices to validate
            underlying_price: Current underlying price

        Returns:
            True if strikes are valid for calendar spread
        """
        if len(strikes) != 1:
            return False

        strike = strikes[0]

        # Strike should be near current price (ATM or slightly OTM)
        price_diff = abs(strike - underlying_price) / underlying_price
        if price_diff > 0.15:  # Within 15% of current price
            return False

        return True


class CalendarPutStrategy(BaseStrategy):
    """
    Calendar Put Spread options strategy implementation.

    Calendar Put Spread consists of:
    - Short put at near expiration (sold for higher time decay)
    - Long put at far expiration with same strike (purchased for time decay protection)

    Structure: Sell Put @ A (near), Buy Put @ A (far)

    Time decay strategy profiting from theta differential between expirations.
    Net debit strategy - pay premium upfront.

    Max profit: When underlying is at strike price at short expiration
    Max loss: Net debit paid (when underlying moves significantly away from strike)
    Breakeven: Strike +/- net debit depending on direction

    Best market conditions:
    - Low volatility regimes (4) for time decay advantage
    - Neutral to slightly bearish bias
    - Avoid high volatility regimes where underlying may move significantly
    """

    def _create_metadata(self) -> StrategyMetadata:
        """Create Calendar Put Spread strategy metadata."""
        return StrategyMetadata(
            name="Calendar Put Spread",
            category=StrategyCategory.ADVANCED,
            risk_level=RiskLevel.MEDIUM,
            capital_requirement=1.2,  # Moderate risk due to time decay complexity
            description="Time decay strategy with short near-term and long far-term puts at same strike",
            typical_market_conditions=[
                "Low volatility market (regime 4)",
                "Neutral to slightly bearish bias",
                "Time decay favorable environment",
                "Stable underlying price expected near strike"
            ]
        )

    def get_strategy_type(self) -> StrategyType:
        """Get strategy type enum."""
        return StrategyType.CALENDAR_PUT

    def validate_market_conditions(self, conditions: MarketConditions) -> bool:
        """
        Validate market conditions for Calendar Put strategy.

        Args:
            conditions: Current market conditions

        Returns:
            True if conditions are suitable for Calendar Put
        """
        # Favorable regime: Low volatility (4)
        if conditions.regime != 4:
            return False

        # Check volatility rank (low volatility preferred)
        if conditions.volatility_rank > 0.5:  # Avoid high volatility
            return False

        # Check trend strength (prefer neutral to slightly bearish)
        if abs(conditions.trend_strength) > 0.6:  # Avoid strong trends
            return False

        # Check time to expiration for near leg (15-45 DTE optimal)
        if conditions.time_to_expiration < 10 or conditions.time_to_expiration > 60:
            return False

        return True

    def calculate_entry_criteria(self, conditions: MarketConditions) -> EntrySignal:
        """
        Calculate entry signal for Calendar Put Spread.

        Args:
            conditions: Current market conditions

        Returns:
            EntrySignal with entry decision and confidence
        """
        reasons = []
        confidence = 0.0

        # Base validation
        if not self.validate_market_conditions(conditions):
            return EntrySignal(
                should_enter=False,
                confidence=0.0,
                reasons=["Market conditions not suitable for calendar spread"]
            )

        # Evaluate entry factors
        # 1. Volatility assessment
        if conditions.volatility_rank < 0.3:
            confidence += 0.3
            reasons.append("Low volatility favors time decay strategy")

        # 2. Trend neutrality
        if abs(conditions.trend_strength) < 0.3:
            confidence += 0.3
            reasons.append("Neutral trend suitable for calendar spread")

        # 3. Time decay window
        if 20 <= conditions.time_to_expiration <= 40:
            confidence += 0.4
            reasons.append("Optimal time to expiration for near leg")

        # Entry decision
        should_enter = confidence >= 0.6

        return EntrySignal(
            should_enter=should_enter,
            confidence=confidence,
            reasons=reasons,
            recommended_size=1.0 if should_enter else 0.0
        )

    def calculate_exit_criteria(self, position: 'Position', conditions: MarketConditions) -> ExitSignal:
        """
        Calculate exit signal for existing Calendar Put position.

        Args:
            position: Current position
            conditions: Current market conditions

        Returns:
            ExitSignal with exit decision and reasoning
        """
        reasons = []
        urgency = 0.0

        # Check for volatility increase
        if conditions.volatility_rank > 0.7:
            urgency += 0.4
            reasons.append("High volatility threatens time decay strategy")

        # Check for strong trend emergence
        if abs(conditions.trend_strength) > 0.7:
            urgency += 0.4
            reasons.append("Strong trend may move underlying away from strike")

        # Check time decay (close when near expiration approaches)
        if conditions.time_to_expiration < 7:
            urgency += 0.5
            reasons.append("Near expiration approaching, time to close")

        # Determine exit decision
        should_exit = urgency >= 0.6
        exit_type = "defensive" if urgency > 0.6 else "time"

        return ExitSignal(
            should_exit=should_exit,
            urgency=urgency,
            exit_type=exit_type,
            reasons=reasons
        )

    def get_risk_metrics(self, strikes: List[int], underlying_price: int,
                        time_to_expiration: int, volatility: float) -> RiskMetrics:
        """
        Calculate risk metrics for Calendar Put Spread.

        Args:
            strikes: Strike prices [strike] (same for both legs)
            underlying_price: Current underlying price in cents
            time_to_expiration: Days to expiration for near leg
            volatility: Implied volatility estimate

        Returns:
            RiskMetrics with comprehensive risk analysis
        """
        if len(strikes) != 1:
            raise ValueError("Calendar spread requires exactly one strike price")

        strike = strikes[0]

        # Estimate near and far leg premiums
        # Near leg (shorter expiration, higher time decay)
        near_premium = max(strike - underlying_price, 0) + int(volatility * strike * 0.1)

        # Far leg (longer expiration, ~2x time)
        far_premium = max(strike - underlying_price, 0) + int(volatility * strike * 0.15)

        # Net debit (cost to enter)
        net_debit = far_premium - near_premium

        # Calculate risk metrics
        # Max profit occurs when stock is at strike at near expiration
        max_profit = int(strike * 0.03)  # Estimate ~3% of strike value

        # Max loss is the net debit paid
        max_loss = net_debit

        # Breakeven points (approximate)
        breakeven_lower = strike - net_debit
        breakeven_upper = strike + net_debit

        # Profit probability (higher when near strike)
        distance_from_strike = abs(underlying_price - strike) / strike
        profit_probability = max(0.3, 0.8 - distance_from_strike * 2)

        # Risk-reward ratio
        risk_reward_ratio = max_profit / max_loss if max_loss > 0 else 0.0

        # Capital requirement is the net debit
        capital_requirement = net_debit

        # Margin requirement (for spread)
        margin_requirement = max(net_debit, int(strike * 0.1))  # 10% of strike minimum

        return RiskMetrics(
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[breakeven_lower, breakeven_upper],
            profit_probability=profit_probability,
            risk_reward_ratio=risk_reward_ratio,
            capital_requirement=capital_requirement,
            margin_requirement=margin_requirement
        )

    def get_position_legs(self, strikes: List[int], expiration_date: datetime) -> List[PositionLeg]:
        """
        Get position legs for Calendar Put Spread construction.

        Args:
            strikes: Strike prices [strike] (same for both legs)
            expiration_date: Near expiration date (far will be calculated)

        Returns:
            List of PositionLeg objects defining the complete position
        """
        if len(strikes) != 1:
            raise ValueError("Calendar spread requires exactly one strike price")

        strike = strikes[0]

        # Calculate far expiration (typically 30-60 days beyond near)
        far_expiration = expiration_date + timedelta(days=35)

        # Create position legs
        legs = [
            # Short near-term put (higher time decay)
            PositionLeg(
                option_type=OptionType.PUT,
                strike=strike,
                quantity=-1,  # Short
                expiration_date=expiration_date
            ),
            # Long far-term put (protection)
            PositionLeg(
                option_type=OptionType.PUT,
                strike=strike,
                quantity=1,   # Long
                expiration_date=far_expiration
            )
        ]

        return legs

    def _validate_strategy_strikes(self, strikes: List[int], underlying_price: int) -> bool:
        """
        Validate strike configuration for Calendar Put Spread.

        Args:
            strikes: Strike prices to validate
            underlying_price: Current underlying price

        Returns:
            True if strikes are valid for calendar spread
        """
        if len(strikes) != 1:
            return False

        strike = strikes[0]

        # Strike should be near current price (ATM or slightly OTM)
        price_diff = abs(strike - underlying_price) / underlying_price
        if price_diff > 0.15:  # Within 15% of current price
            return False

        return True
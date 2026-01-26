"""
Equity-based options strategies implementation.

Implements strategies that require equity positions combined with options overlays,
including Covered Calls and Collar strategies for income generation and protection.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

from .base import (
    BaseStrategy, StrategyMetadata, StrategyCategory, RiskLevel,
    MarketConditions, EntrySignal, ExitSignal, RiskMetrics, PositionLeg,
    OptionType, StrategyType
)

if TYPE_CHECKING:
    from ..features.position_models import Position


@dataclass
class EquityLeg:
    """
    Equity position leg for strategies requiring stock positions.

    Attributes:
        symbol: Stock symbol
        quantity: Number of shares (positive for long)
        price: Entry price per share in cents
    """
    symbol: str
    quantity: int
    price: int


class CoveredCallStrategy(BaseStrategy):
    """
    Covered Call options strategy implementation.

    Covered Call consists of:
    - Long 100 shares of underlying stock
    - Short 1 call option (typically OTM)

    Structure: Long 100 shares + Short Call @ A (A > current price)

    Income generation strategy with upside capping.
    Reduces cost basis of stock position through premium collection.

    Max profit: Premium received + (Strike - Stock Price) if assigned
    Max loss: Stock price decline minus premium received (substantial downside risk)
    Breakeven: Stock price - premium received

    Best market conditions:
    - Sideways to slightly bullish regimes (1, 5)
    - Moderate volatility for premium collection
    - Existing stock position or bullish outlook on stock
    """

    def _create_metadata(self) -> StrategyMetadata:
        """Create Covered Call strategy metadata."""
        return StrategyMetadata(
            name="Covered Call",
            category=StrategyCategory.ADVANCED,
            risk_level=RiskLevel.MEDIUM,
            capital_requirement=2.5,  # Requires full stock position capital
            description="Income strategy combining long stock position with short call for premium collection",
            typical_market_conditions=[
                "Sideways market (regime 5)",
                "Slightly bullish trend (regime 1)",
                "Moderate to high volatility for premium",
                "Existing stock position or bullish outlook"
            ]
        )

    def get_strategy_type(self) -> StrategyType:
        """Get strategy type enum."""
        return StrategyType.LONG_CALL  # Using LONG_CALL as placeholder since COVERED_CALL not in enum

    def validate_market_conditions(self, conditions: MarketConditions) -> bool:
        """
        Validate market conditions for Covered Call strategy.

        Args:
            conditions: Current market conditions

        Returns:
            True if conditions are suitable for Covered Call
        """
        # Favorable regimes: Bull trending (1) and Sideways (5)
        favorable_regimes = [1, 5]

        # Check regime suitability
        if conditions.regime not in favorable_regimes:
            return False

        # Check trend strength (neutral to moderately bullish)
        if conditions.trend_strength < -0.3 or conditions.trend_strength > 0.7:
            return False

        # Check volatility for premium collection (moderate to high)
        if conditions.volatility_rank < 0.3:
            return False

        # Check time to expiration (30-60 DTE optimal for premium decay)
        if conditions.time_to_expiration < 20 or conditions.time_to_expiration > 90:
            return False

        return True

    def calculate_entry_criteria(self, conditions: MarketConditions) -> EntrySignal:
        """
        Calculate entry signal for Covered Call.

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
                reasons=["Market conditions not suitable for covered call"]
            )

        # Evaluate entry factors
        # 1. Volatility assessment for premium collection
        if conditions.volatility_rank > 0.5:
            confidence += 0.4
            reasons.append("High volatility provides attractive call premiums")

        # 2. Trend assessment (neutral to slightly bullish)
        if 0.0 <= conditions.trend_strength <= 0.4:
            confidence += 0.3
            reasons.append("Neutral to slightly bullish trend suitable for covered calls")

        # 3. Regime assessment
        if conditions.regime in [1, 5]:
            confidence += 0.3
            reasons.append("Favorable regime for covered call strategy")

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
        Calculate exit signal for existing Covered Call position.

        Args:
            position: Current position
            conditions: Current market conditions

        Returns:
            ExitSignal with exit decision and reasoning
        """
        reasons = []
        urgency = 0.0

        # Check for strong bearish trend (threatens stock position)
        if conditions.trend_strength < -0.6:
            urgency += 0.5
            reasons.append("Strong bearish trend threatens underlying stock position")

        # Check for low volatility (reduces future premium potential)
        if conditions.volatility_rank < 0.2:
            urgency += 0.3
            reasons.append("Low volatility reduces future call premium potential")

        # Check for assignment risk (very ITM calls)
        # This would need position data to evaluate properly
        if conditions.time_to_expiration < 10:
            urgency += 0.4
            reasons.append("Approaching expiration, evaluate assignment risk")

        # Determine exit decision
        should_exit = urgency >= 0.6
        exit_type = "defensive" if urgency > 0.6 else "management"

        return ExitSignal(
            should_exit=should_exit,
            urgency=urgency,
            exit_type=exit_type,
            reasons=reasons
        )

    def get_risk_metrics(self, strikes: List[int], underlying_price: int,
                        time_to_expiration: int, volatility: float) -> RiskMetrics:
        """
        Calculate risk metrics for Covered Call.

        Args:
            strikes: Strike prices [call_strike]
            underlying_price: Current underlying price in cents
            time_to_expiration: Days to expiration
            volatility: Implied volatility estimate

        Returns:
            RiskMetrics with comprehensive risk analysis
        """
        if len(strikes) != 1:
            raise ValueError("Covered call requires exactly one strike price (call strike)")

        call_strike = strikes[0]

        # Estimate call premium (time value + intrinsic value)
        intrinsic_value = max(underlying_price - call_strike, 0)
        time_value = int(volatility * call_strike * 0.1 * (time_to_expiration / 365)**0.5)
        call_premium = intrinsic_value + time_value

        # Calculate risk metrics
        # Max profit: Call premium + (Strike - Stock Price) if assigned
        if call_strike > underlying_price:
            max_profit = call_premium + (call_strike - underlying_price)
        else:
            max_profit = call_premium  # Already ITM, just collect premium

        # Max loss: Stock can go to zero minus premium collected
        max_loss = underlying_price - call_premium

        # Breakeven: Stock price minus premium received
        breakeven = underlying_price - call_premium

        # Profit probability (depends on stock staying below strike)
        if call_strike > underlying_price:
            profit_probability = 0.7  # Estimate based on OTM calls
        else:
            profit_probability = 0.9  # ITM calls almost certainly assigned but still profitable

        # Risk-reward ratio
        risk_reward_ratio = max_profit / max_loss if max_loss > 0 else 0.0

        # Capital requirement: Full stock price (100 shares)
        capital_requirement = underlying_price * 100

        # Margin requirement: Stock value minus call premium collected
        margin_requirement = capital_requirement - call_premium * 100

        return RiskMetrics(
            max_profit=max_profit * 100,  # 100 shares
            max_loss=max_loss * 100,      # 100 shares
            breakeven_points=[breakeven],
            profit_probability=profit_probability,
            risk_reward_ratio=risk_reward_ratio,
            capital_requirement=capital_requirement,
            margin_requirement=margin_requirement
        )

    def get_position_legs(self, strikes: List[int], expiration_date: datetime) -> List[PositionLeg]:
        """
        Get position legs for Covered Call construction.

        Args:
            strikes: Strike prices [call_strike]
            expiration_date: Option expiration date

        Returns:
            List of PositionLeg objects defining the complete position
        """
        if len(strikes) != 1:
            raise ValueError("Covered call requires exactly one strike price")

        call_strike = strikes[0]

        # Create position legs
        legs = [
            # Short call option
            PositionLeg(
                option_type=OptionType.CALL,
                strike=call_strike,
                quantity=-1,  # Short (sell call)
                expiration_date=expiration_date
            )
        ]

        # Note: The equity position (100 shares) would be handled separately
        # in a complete implementation with EquityLeg objects

        return legs

    def _validate_strategy_strikes(self, strikes: List[int], underlying_price: int) -> bool:
        """
        Validate strike configuration for Covered Call.

        Args:
            strikes: Strike prices to validate
            underlying_price: Current underlying price

        Returns:
            True if strikes are valid for covered call
        """
        if len(strikes) != 1:
            return False

        call_strike = strikes[0]

        # Call strike should be at or above current price
        if call_strike < underlying_price * 0.95:  # Allow slightly ITM
            return False

        # Strike shouldn't be too far OTM (reduces premium)
        if call_strike > underlying_price * 1.2:  # Within 20% OTM
            return False

        return True


class CollarStrategy(BaseStrategy):
    """
    Collar options strategy implementation.

    Collar consists of:
    - Long 100 shares of underlying stock
    - Long 1 put option (OTM protective put)
    - Short 1 call option (OTM covered call)

    Structure: Long 100 shares + Long Put @ A + Short Call @ B (A < current < B)

    Protective strategy with income generation.
    Limits both upside and downside while generating income.

    Max profit: (Call Strike - Stock Price) + Call Premium - Put Premium
    Max loss: (Stock Price - Put Strike) + Call Premium - Put Premium
    Breakeven: Stock Price + Net Premium Paid/Received

    Best market conditions:
    - Uncertain or protective scenarios (regimes 3, 7, 8)
    - Volatile markets where protection is valued
    - When holding stock and wanting to limit risk
    """

    def _create_metadata(self) -> StrategyMetadata:
        """Create Collar strategy metadata."""
        return StrategyMetadata(
            name="Collar",
            category=StrategyCategory.ADVANCED,
            risk_level=RiskLevel.LOW,
            capital_requirement=2.3,  # Requires stock position but limited risk
            description="Protective strategy combining stock, protective put, and covered call for limited risk/reward",
            typical_market_conditions=[
                "Uncertain market conditions (regime 3)",
                "High volatility periods (regimes 7, 8)",
                "When portfolio protection is prioritized",
                "Existing stock positions requiring protection"
            ]
        )

    def get_strategy_type(self) -> StrategyType:
        """Get strategy type enum."""
        return StrategyType.LONG_PUT  # Using LONG_PUT as placeholder since COLLAR not in enum

    def validate_market_conditions(self, conditions: MarketConditions) -> bool:
        """
        Validate market conditions for Collar strategy.

        Args:
            conditions: Current market conditions

        Returns:
            True if conditions are suitable for Collar
        """
        # Favorable regimes: Uncertain (3), High volatility (7, 8)
        favorable_regimes = [3, 7, 8]

        # Check regime suitability
        if conditions.regime not in favorable_regimes:
            return False

        # Check volatility (higher volatility makes protection more valuable)
        if conditions.volatility_rank < 0.4:
            return False

        # Check time to expiration (30-90 DTE for protective strategies)
        if conditions.time_to_expiration < 25 or conditions.time_to_expiration > 120:
            return False

        return True

    def calculate_entry_criteria(self, conditions: MarketConditions) -> EntrySignal:
        """
        Calculate entry signal for Collar.

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
                reasons=["Market conditions not suitable for collar strategy"]
            )

        # Evaluate entry factors
        # 1. Volatility assessment (higher volatility increases protection value)
        if conditions.volatility_rank > 0.6:
            confidence += 0.4
            reasons.append("High volatility makes protective collar more attractive")

        # 2. Regime assessment (uncertainty favors protection)
        if conditions.regime in [3, 7, 8]:
            confidence += 0.4
            reasons.append("Uncertain market conditions favor protective strategies")

        # 3. Trend uncertainty
        if abs(conditions.trend_strength) < 0.5:
            confidence += 0.2
            reasons.append("Uncertain trend direction supports collar strategy")

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
        Calculate exit signal for existing Collar position.

        Args:
            position: Current position
            conditions: Current market conditions

        Returns:
            ExitSignal with exit decision and reasoning
        """
        reasons = []
        urgency = 0.0

        # Check for low volatility (reduces value of protection)
        if conditions.volatility_rank < 0.3:
            urgency += 0.3
            reasons.append("Low volatility reduces value of protective collar")

        # Check for strong trend establishment
        if abs(conditions.trend_strength) > 0.7:
            urgency += 0.3
            reasons.append("Strong trend reduces need for protective collar")

        # Check for approaching expiration
        if conditions.time_to_expiration < 15:
            urgency += 0.4
            reasons.append("Approaching expiration, consider rolling protection")

        # Determine exit decision
        should_exit = urgency >= 0.6
        exit_type = "rebalance" if urgency > 0.6 else "hold"

        return ExitSignal(
            should_exit=should_exit,
            urgency=urgency,
            exit_type=exit_type,
            reasons=reasons
        )

    def get_risk_metrics(self, strikes: List[int], underlying_price: int,
                        time_to_expiration: int, volatility: float) -> RiskMetrics:
        """
        Calculate risk metrics for Collar.

        Args:
            strikes: Strike prices [put_strike, call_strike]
            underlying_price: Current underlying price in cents
            time_to_expiration: Days to expiration
            volatility: Implied volatility estimate

        Returns:
            RiskMetrics with comprehensive risk analysis
        """
        if len(strikes) != 2:
            raise ValueError("Collar requires exactly two strike prices [put_strike, call_strike]")

        put_strike, call_strike = strikes

        # Estimate option premiums
        put_premium = max(put_strike - underlying_price, 0) + int(volatility * put_strike * 0.08)
        call_premium = max(underlying_price - call_strike, 0) + int(volatility * call_strike * 0.08)

        # Net premium (call premium received minus put premium paid)
        net_premium = call_premium - put_premium

        # Calculate risk metrics
        # Max profit: (Call Strike - Stock Price) + Net Premium
        max_profit = (call_strike - underlying_price) + net_premium

        # Max loss: (Stock Price - Put Strike) - Net Premium
        max_loss = (underlying_price - put_strike) - net_premium

        # Breakeven: Stock price adjusted by net premium
        if net_premium >= 0:  # Net credit
            breakeven = underlying_price - net_premium
        else:  # Net debit
            breakeven = underlying_price + abs(net_premium)

        # Profit probability (depends on staying between strikes)
        profit_probability = 0.6  # Conservative estimate for range-bound outcome

        # Risk-reward ratio
        risk_reward_ratio = max_profit / abs(max_loss) if max_loss != 0 else 0.0

        # Capital requirement: Stock value plus put premium (call premium offsets)
        capital_requirement = underlying_price * 100 + max(0, put_premium - call_premium) * 100

        # Margin requirement: Reduced due to protective nature
        margin_requirement = max(capital_requirement * 50 // 100, underlying_price * 20)  # 50% or 20% minimum

        return RiskMetrics(
            max_profit=max_profit * 100,  # 100 shares
            max_loss=abs(max_loss) * 100,  # 100 shares
            breakeven_points=[breakeven],
            profit_probability=profit_probability,
            risk_reward_ratio=risk_reward_ratio,
            capital_requirement=capital_requirement,
            margin_requirement=margin_requirement
        )

    def get_position_legs(self, strikes: List[int], expiration_date: datetime) -> List[PositionLeg]:
        """
        Get position legs for Collar construction.

        Args:
            strikes: Strike prices [put_strike, call_strike]
            expiration_date: Option expiration date

        Returns:
            List of PositionLeg objects defining the complete position
        """
        if len(strikes) != 2:
            raise ValueError("Collar requires exactly two strike prices")

        put_strike, call_strike = strikes

        # Create position legs
        legs = [
            # Long protective put
            PositionLeg(
                option_type=OptionType.PUT,
                strike=put_strike,
                quantity=1,   # Long (buy put for protection)
                expiration_date=expiration_date
            ),
            # Short covered call
            PositionLeg(
                option_type=OptionType.CALL,
                strike=call_strike,
                quantity=-1,  # Short (sell call for income)
                expiration_date=expiration_date
            )
        ]

        # Note: The equity position (100 shares) would be handled separately
        # in a complete implementation with EquityLeg objects

        return legs

    def _validate_strategy_strikes(self, strikes: List[int], underlying_price: int) -> bool:
        """
        Validate strike configuration for Collar.

        Args:
            strikes: Strike prices to validate [put_strike, call_strike]
            underlying_price: Current underlying price

        Returns:
            True if strikes are valid for collar
        """
        if len(strikes) != 2:
            return False

        put_strike, call_strike = strikes

        # Put strike should be below current price (OTM protective put)
        if put_strike >= underlying_price:
            return False

        # Call strike should be above current price (OTM covered call)
        if call_strike <= underlying_price:
            return False

        # Ensure reasonable spread (put should not be too far OTM)
        if put_strike < underlying_price * 0.8:  # Put within 20% of current
            return False

        # Call should not be too far OTM (reduces premium)
        if call_strike > underlying_price * 1.2:  # Call within 20% of current
            return False

        return True
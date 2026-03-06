"""
Directional options strategies implementation.

Implements Bull/Bear Call/Put spread strategies for trending market conditions.
These strategies profit from directional price movements with limited risk profiles.
"""

from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

from .base import (
    BaseStrategy, StrategyMetadata, StrategyCategory, RiskLevel,
    MarketConditions, EntrySignal, ExitSignal, RiskMetrics, PositionLeg,
    OptionType, StrategyType
)

if TYPE_CHECKING:
    from ..features.position_models import Position


class BullCallSpreadStrategy(BaseStrategy):
    """
    Bull Call Spread options strategy implementation.

    Bull Call Spread consists of:
    - Long call at lower strike (purchased)
    - Short call at higher strike (sold)

    Structure: Buy Call @ A, Sell Call @ B (A < B)

    Bullish strategy with limited upside and limited downside.
    Net debit strategy - pay premium upfront.

    Max profit: (B - A) - net debit paid
    Max loss: Net debit paid
    Breakeven: A + net debit paid

    Best market conditions:
    - Bull trending regimes (1) and Recovery regimes (6)
    - Moderate volatility
    - Expect upward price movement but not unlimited
    """

    def _create_metadata(self) -> StrategyMetadata:
        """Create Bull Call Spread strategy metadata."""
        return StrategyMetadata(
            name="Bull Call Spread",
            category=StrategyCategory.DIRECTIONAL,
            risk_level=RiskLevel.LOW,
            capital_requirement=1.0,  # Limited risk due to debit spread nature
            description="Bullish debit spread with limited upside and downside for moderate upward moves",
            typical_market_conditions=[
                "Bull trending market (regime 1)",
                "Recovery phase (regime 6)",
                "Moderate volatility environment",
                "Expected moderate upward price movement"
            ]
        )

    def get_strategy_type(self) -> StrategyType:
        """Get strategy type enum."""
        return StrategyType.BULL_CALL_SPREAD

    def validate_market_conditions(self, conditions: MarketConditions) -> bool:
        """
        Validate market conditions for Bull Call Spread strategy.

        Args:
            conditions: Current market conditions

        Returns:
            True if conditions are suitable for Bull Call Spread
        """
        # Favorable regimes: Bull trending (1) and Recovery (6)
        favorable_regimes = [1, 6]

        # Check regime suitability
        if conditions.regime not in favorable_regimes:
            return False

        # Check trend strength (should be bullish)
        if conditions.trend_strength < 0.2:  # Need some upward bias
            return False

        # Check volatility (moderate levels work best)
        if conditions.volatility_rank < 0.2 or conditions.volatility_rank > 0.8:
            return False

        # Check time to expiration (30-60 DTE optimal)
        if conditions.time_to_expiration < 20 or conditions.time_to_expiration > 90:
            return False

        return True

    def calculate_entry_criteria(self, conditions: MarketConditions) -> EntrySignal:
        """
        Calculate entry signal for Bull Call Spread.

        Args:
            conditions: Current market conditions

        Returns:
            EntrySignal with entry decision and confidence
        """
        if not self.validate_market_conditions(conditions):
            return EntrySignal(
                should_enter=False,
                confidence=0.0,
                reasons=["Market conditions not suitable for Bull Call Spread"]
            )

        confidence = 0.0
        reasons = []

        # Regime scoring
        if conditions.regime == 1:  # Bull trending - perfect
            confidence += 0.35
            reasons.append("Bull trending regime ideal for bull call spread")
        elif conditions.regime == 6:  # Recovery - good
            confidence += 0.25
            reasons.append("Recovery regime favorable for bullish strategies")

        # Trend strength scoring
        if conditions.trend_strength >= 0.6:  # Strong bullish trend
            confidence += 0.25
            reasons.append(f"Strong bullish trend {conditions.trend_strength:.2f} supports call spread")
        elif conditions.trend_strength >= 0.4:  # Moderate bullish trend
            confidence += 0.2
            reasons.append("Moderate bullish trend supports strategy")
        elif conditions.trend_strength >= 0.2:  # Weak bullish trend
            confidence += 0.1
            reasons.append("Weak bullish bias present")

        # Volatility scoring (moderate is best)
        if 0.4 <= conditions.volatility_rank <= 0.6:  # Sweet spot
            confidence += 0.2
            reasons.append("Optimal volatility level for call spreads")
        elif 0.2 <= conditions.volatility_rank < 0.4:  # Lower but acceptable
            confidence += 0.15
            reasons.append("Lower volatility acceptable for debit spreads")
        elif 0.6 < conditions.volatility_rank <= 0.8:  # Higher but manageable
            confidence += 0.1
            reasons.append("Higher volatility increases premium cost")

        # Time to expiration scoring
        if 30 <= conditions.time_to_expiration <= 60:  # Optimal range
            confidence += 0.2
            reasons.append(f"Optimal {conditions.time_to_expiration} DTE for call spread")
        elif 20 <= conditions.time_to_expiration < 30:  # Shorter term
            confidence += 0.15
            reasons.append("Shorter time frame acceptable")
        elif 60 < conditions.time_to_expiration <= 90:  # Longer term
            confidence += 0.1
            reasons.append("Longer timeframe reduces time decay benefit")

        # Position sizing recommendation
        recommended_size = 0.9
        if confidence > 0.8:
            recommended_size = 1.25  # Increase size for high confidence
        elif confidence < 0.5:
            recommended_size = 0.75  # Reduce size for lower confidence

        return EntrySignal(
            should_enter=confidence >= 0.6,
            confidence=confidence,
            reasons=reasons,
            recommended_size=recommended_size
        )

    def calculate_exit_criteria(self, position: 'Position', conditions: MarketConditions) -> ExitSignal:
        """
        Calculate exit signal for existing Bull Call Spread position.

        Args:
            position: Current Bull Call Spread position
            conditions: Current market conditions

        Returns:
            ExitSignal with exit decision and reasoning
        """
        reasons = []
        urgency = 0.0
        exit_type = ""

        # Time-based exit
        days_to_exp = position.days_to_expiration
        if days_to_exp <= 5:
            urgency += 0.7
            exit_type = "time"
            reasons.append("Approaching expiration - high gamma risk")
        elif days_to_exp <= 10:
            urgency += 0.3
            reasons.append("Short time to expiration")

        # Profit target exit
        current_pnl = position.calculate_unrealized_pnl()
        max_profit = position.calculate_max_profit()

        if max_profit > 0:
            profit_percent = current_pnl / max_profit
            if profit_percent >= 0.75:  # 75% of max profit
                urgency += 0.8
                exit_type = "profit"
                reasons.append(f"Excellent profit target reached: {profit_percent:.1%}")
            elif profit_percent >= 0.5:  # 50% of max profit
                urgency += 0.5
                exit_type = "profit"
                reasons.append(f"Good profit level: {profit_percent:.1%}")
            elif profit_percent >= 0.25:  # 25% profit consideration
                urgency += 0.2
                if not exit_type:
                    exit_type = "profit"
                reasons.append(f"Partial profit available: {profit_percent:.1%}")

        # Loss limit exit
        max_loss = position.calculate_max_loss()
        if max_loss > 0:
            loss_percent = abs(current_pnl) / max_loss
            if current_pnl < 0 and loss_percent >= 0.8:  # 80% of max loss
                urgency = 1.0
                exit_type = "loss"
                reasons.append("Approaching maximum loss threshold")
            elif current_pnl < 0 and loss_percent >= 0.5:  # 50% loss
                urgency += 0.6
                exit_type = "loss"
                reasons.append("Significant loss incurred")

        # Market regime change
        unfavorable_regimes = [2, 3, 7, 8]  # Bear trending, high vol, distribution, crisis
        if conditions.regime in unfavorable_regimes:
            urgency += 0.4
            if not exit_type:
                exit_type = "adjustment"
            reasons.append(f"Market regime {conditions.regime} unfavorable for bull strategies")

        # Trend reversal
        if conditions.trend_strength < -0.2:  # Turned bearish
            urgency += 0.6
            if not exit_type:
                exit_type = "adjustment"
            reasons.append("Trend has turned bearish")
        elif conditions.trend_strength < 0.1:  # Lost bullish bias
            urgency += 0.3
            if not exit_type:
                exit_type = "adjustment"
            reasons.append("Bullish momentum weakening")

        # Volatility expansion (can help if in-the-money)
        if conditions.volatility_rank > 0.9:
            if current_pnl > 0:
                urgency += 0.2
                reasons.append("High volatility - consider profit taking")
            else:
                urgency += 0.4
                reasons.append("High volatility may increase losses")

        # Price target reached (near upper strike)
        breakevens = position.calculate_breakevens()
        current_price = position.current_underlying_price

        if len(position.strikes) >= 2:
            upper_strike = max(position.strikes)
            lower_strike = min(position.strikes)

            # Check if price is near upper strike (profit zone)
            distance_to_upper = (upper_strike - current_price) / upper_strike
            if distance_to_upper < 0.05:  # Within 5% of upper strike
                urgency += 0.6
                exit_type = "profit"
                reasons.append("Price approaching maximum profit zone")

            # Check if price fell below lower strike
            if current_price < lower_strike:
                urgency += 0.5
                if not exit_type:
                    exit_type = "loss"
                reasons.append("Price below long call strike")

        return ExitSignal(
            should_exit=urgency >= 0.6,
            urgency=min(urgency, 1.0),
            exit_type=exit_type or "hold",
            reasons=reasons or ["Position within acceptable parameters"]
        )

    def get_risk_metrics(self, strikes: List[int], underlying_price: int,
                        time_to_expiration: int, volatility: float) -> RiskMetrics:
        """
        Calculate risk metrics for Bull Call Spread.

        Args:
            strikes: [long_call_strike, short_call_strike] in cents
            underlying_price: Current underlying price in cents
            time_to_expiration: Days to expiration
            volatility: Implied volatility

        Returns:
            RiskMetrics for the Bull Call Spread
        """
        if len(strikes) != 2:
            raise ValueError("Bull Call Spread requires exactly 2 strikes")

        long_strike, short_strike = strikes

        if long_strike >= short_strike:
            raise ValueError("Long strike must be lower than short strike")

        # Calculate spread width
        spread_width = short_strike - long_strike

        # Estimate net debit (simplified - would use Black-Scholes in production)
        # Assume roughly 30-40% of spread width as net debit for ATM spreads
        atm_factor = abs(underlying_price - long_strike) / underlying_price
        debit_factor = 0.35 - (atm_factor * 0.1)  # Reduce debit for OTM spreads
        net_debit = int(spread_width * max(debit_factor, 0.15))

        # Risk calculations
        max_profit = spread_width - net_debit
        max_loss = net_debit

        # Breakeven point
        breakeven = long_strike + net_debit
        breakevens = [breakeven]

        # Probability estimation
        # Simple model: probability price ends above breakeven
        price_move_needed = (breakeven - underlying_price) / underlying_price
        # Rough approximation based on normal distribution
        if price_move_needed <= 0:
            profit_probability = 0.7  # Already ITM
        else:
            profit_probability = max(0.3, 0.6 - (price_move_needed * 2))

        # Risk/reward ratio
        risk_reward_ratio = max_profit / max_loss if max_loss > 0 else float('inf')

        # Capital requirements
        capital_requirement = max_loss  # Net debit paid
        margin_requirement = int(max_loss * 1.1)  # Small buffer for debit spreads

        return RiskMetrics(
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=breakevens,
            profit_probability=profit_probability,
            risk_reward_ratio=risk_reward_ratio,
            capital_requirement=capital_requirement,
            margin_requirement=margin_requirement
        )

    def get_position_legs(self, strikes: List[int], expiration_date: datetime) -> List[PositionLeg]:
        """
        Get position legs for Bull Call Spread construction.

        Args:
            strikes: [long_call_strike, short_call_strike] in cents
            expiration_date: Option expiration date

        Returns:
            List of PositionLeg objects for Bull Call Spread
        """
        if len(strikes) != 2:
            raise ValueError("Bull Call Spread requires exactly 2 strikes: [long_call, short_call]")

        long_call_strike, short_call_strike = strikes

        return [
            # Long call at lower strike
            PositionLeg(
                option_type=OptionType.CALL,
                strike=long_call_strike,
                quantity=1,  # Long position
                expiration_date=expiration_date
            ),
            # Short call at higher strike
            PositionLeg(
                option_type=OptionType.CALL,
                strike=short_call_strike,
                quantity=-1,  # Short position
                expiration_date=expiration_date
            )
        ]

    def _validate_strategy_strikes(self, strikes: List[int], underlying_price: int) -> bool:
        """
        Validate Bull Call Spread strike configuration.

        Args:
            strikes: Strike prices to validate
            underlying_price: Current underlying price

        Returns:
            True if strikes are valid for Bull Call Spread
        """
        if len(strikes) != 2:
            return False

        long_strike, short_strike = strikes

        # Long strike must be lower than short strike
        if long_strike >= short_strike:
            return False

        # Spread width should be reasonable ($1 to $20)
        spread_width = short_strike - long_strike
        if spread_width < 100 or spread_width > 2000:  # $1 to $20
            return False

        # Long strike should be reasonably close to current price (within 20%)
        distance_from_price = abs(long_strike - underlying_price) / underlying_price
        if distance_from_price > 0.20:
            return False

        # Short strike should be above current price for bullish strategy
        if short_strike <= underlying_price:
            return False

        return True


class BearCallSpreadStrategy(BaseStrategy):
    """
    Bear Call Spread options strategy implementation.

    Bear Call Spread consists of:
    - Short call at lower strike (sold)
    - Long call at higher strike (purchased)

    Structure: Sell Call @ A, Buy Call @ B (A < B)

    Bearish/neutral strategy with limited upside and limited downside.
    Net credit strategy - receive premium upfront.

    Max profit: Net credit received
    Max loss: (B - A) - net credit received
    Breakeven: A + net credit received

    Best market conditions:
    - Bear trending regimes (2) and Distribution regimes (7)
    - Expect sideways to downward price movement
    - High implied volatility that can contract
    """

    def _create_metadata(self) -> StrategyMetadata:
        """Create Bear Call Spread strategy metadata."""
        return StrategyMetadata(
            name="Bear Call Spread",
            category=StrategyCategory.DIRECTIONAL,
            risk_level=RiskLevel.MEDIUM,
            capital_requirement=1.2,  # Higher than bull call due to margin requirements
            description="Bearish credit spread profiting from sideways/downward moves with limited risk",
            typical_market_conditions=[
                "Bear trending market (regime 2)",
                "Distribution phase (regime 7)",
                "High implied volatility",
                "Expected sideways to downward price movement"
            ]
        )

    def get_strategy_type(self) -> StrategyType:
        """Get strategy type enum."""
        return StrategyType.BEAR_CALL_SPREAD

    def validate_market_conditions(self, conditions: MarketConditions) -> bool:
        """
        Validate market conditions for Bear Call Spread strategy.

        Args:
            conditions: Current market conditions

        Returns:
            True if conditions are suitable for Bear Call Spread
        """
        # Favorable regimes: Bear trending (2) and Distribution (7)
        favorable_regimes = [2, 7]

        # Check regime suitability
        if conditions.regime not in favorable_regimes:
            return False

        # Check trend strength (should be bearish or neutral)
        if conditions.trend_strength > 0.2:  # Too bullish
            return False

        # Check volatility (higher levels work better for credit collection)
        if conditions.volatility_rank < 0.3:
            return False

        # Check time to expiration (20-60 DTE optimal)
        if conditions.time_to_expiration < 15 or conditions.time_to_expiration > 90:
            return False

        return True

    def calculate_entry_criteria(self, conditions: MarketConditions) -> EntrySignal:
        """
        Calculate entry signal for Bear Call Spread.

        Args:
            conditions: Current market conditions

        Returns:
            EntrySignal with entry decision and confidence
        """
        if not self.validate_market_conditions(conditions):
            return EntrySignal(
                should_enter=False,
                confidence=0.0,
                reasons=["Market conditions not suitable for Bear Call Spread"]
            )

        confidence = 0.0
        reasons = []

        # Regime scoring
        if conditions.regime == 2:  # Bear trending - perfect
            confidence += 0.35
            reasons.append("Bear trending regime ideal for bear call spread")
        elif conditions.regime == 7:  # Distribution - good
            confidence += 0.25
            reasons.append("Distribution regime favorable for bearish strategies")

        # Trend strength scoring (bearish is best)
        if conditions.trend_strength <= -0.4:  # Strong bearish trend
            confidence += 0.25
            reasons.append(f"Strong bearish trend {conditions.trend_strength:.2f} supports call spread")
        elif conditions.trend_strength <= -0.2:  # Moderate bearish trend
            confidence += 0.2
            reasons.append("Moderate bearish trend supports strategy")
        elif conditions.trend_strength <= 0.2:  # Neutral/sideways
            confidence += 0.15
            reasons.append("Neutral trend acceptable for credit spread")

        # Volatility scoring (higher is better for credit collection)
        if conditions.volatility_rank >= 0.7:  # High volatility
            confidence += 0.25
            reasons.append("High volatility excellent for credit collection")
        elif conditions.volatility_rank >= 0.5:  # Moderate-high volatility
            confidence += 0.2
            reasons.append("Good volatility level for premium collection")
        elif conditions.volatility_rank >= 0.3:  # Moderate volatility
            confidence += 0.1
            reasons.append("Moderate volatility acceptable")

        # Time to expiration scoring
        if 30 <= conditions.time_to_expiration <= 50:  # Optimal range
            confidence += 0.15
            reasons.append(f"Optimal {conditions.time_to_expiration} DTE for credit spread")
        elif 20 <= conditions.time_to_expiration < 30:  # Shorter term
            confidence += 0.1
            reasons.append("Shorter timeframe increases time decay benefit")
        elif 50 < conditions.time_to_expiration <= 90:  # Longer term
            confidence += 0.05
            reasons.append("Longer timeframe acceptable for credit collection")

        # Position sizing recommendation
        recommended_size = 1.0
        if confidence > 0.8:
            recommended_size = 1.2  # Moderate increase for high confidence
        elif confidence < 0.5:
            recommended_size = 0.8  # Conservative sizing for lower confidence

        return EntrySignal(
            should_enter=confidence >= 0.6,
            confidence=confidence,
            reasons=reasons,
            recommended_size=recommended_size
        )

    def calculate_exit_criteria(self, position: 'Position', conditions: MarketConditions) -> ExitSignal:
        """
        Calculate exit signal for existing Bear Call Spread position.

        Args:
            position: Current Bear Call Spread position
            conditions: Current market conditions

        Returns:
            ExitSignal with exit decision and reasoning
        """
        reasons = []
        urgency = 0.0
        exit_type = ""

        # Time-based exit
        days_to_exp = position.days_to_expiration
        if days_to_exp <= 7:
            urgency += 0.6
            exit_type = "time"
            reasons.append("Approaching expiration - assignment risk for short call")
        elif days_to_exp <= 15:
            urgency += 0.2
            reasons.append("Short time to expiration")

        # Profit target exit (credit spreads)
        current_pnl = position.calculate_unrealized_pnl()
        max_profit = position.calculate_max_profit()

        if max_profit > 0:
            profit_percent = current_pnl / max_profit
            if profit_percent >= 0.6:  # 60% of max profit
                urgency += 0.7
                exit_type = "profit"
                reasons.append(f"Excellent profit target reached: {profit_percent:.1%}")
            elif profit_percent >= 0.4:  # 40% of max profit
                urgency += 0.4
                exit_type = "profit"
                reasons.append(f"Good profit level: {profit_percent:.1%}")
            elif profit_percent >= 0.2:  # 20% profit consideration
                urgency += 0.2
                if not exit_type:
                    exit_type = "profit"
                reasons.append(f"Partial profit available: {profit_percent:.1%}")

        # Loss limit exit
        max_loss = position.calculate_max_loss()
        if max_loss > 0:
            loss_percent = abs(current_pnl) / max_loss
            if current_pnl < 0 and loss_percent >= 0.75:  # 75% of max loss
                urgency = 1.0
                exit_type = "loss"
                reasons.append("Approaching maximum loss threshold")
            elif current_pnl < 0 and loss_percent >= 0.5:  # 50% loss
                urgency += 0.7
                exit_type = "loss"
                reasons.append("Significant loss incurred")

        # Market regime change
        unfavorable_regimes = [1, 3, 6, 8]  # Bull trending, high vol, recovery, crisis
        if conditions.regime in unfavorable_regimes:
            urgency += 0.4
            if not exit_type:
                exit_type = "adjustment"
            reasons.append(f"Market regime {conditions.regime} unfavorable for bear strategies")

        # Trend reversal (turned bullish)
        if conditions.trend_strength > 0.3:  # Strong bullish turn
            urgency += 0.7
            if not exit_type:
                exit_type = "adjustment"
            reasons.append("Strong bullish trend threatens bear call spread")
        elif conditions.trend_strength > 0.1:  # Moderate bullish turn
            urgency += 0.4
            if not exit_type:
                exit_type = "adjustment"
            reasons.append("Bullish momentum developing")

        # Volatility crush (bad for credit spreads if need to buy back)
        if conditions.volatility_rank < 0.2 and current_pnl < 0:
            urgency += 0.3
            reasons.append("Low volatility makes position adjustment expensive")

        # Price movement above short strike (assignment risk)
        if len(position.strikes) >= 2:
            short_strike = min(position.strikes)  # Lower strike is the short call
            current_price = position.current_underlying_price

            if current_price >= short_strike:
                urgency += 0.6
                if not exit_type:
                    exit_type = "loss"
                reasons.append("Price above short call strike - high assignment risk")
            elif current_price > short_strike * 0.95:  # Within 5% of short strike
                urgency += 0.3
                reasons.append("Price approaching short call strike")

        return ExitSignal(
            should_exit=urgency >= 0.6,
            urgency=min(urgency, 1.0),
            exit_type=exit_type or "hold",
            reasons=reasons or ["Position within acceptable parameters"]
        )

    def get_risk_metrics(self, strikes: List[int], underlying_price: int,
                        time_to_expiration: int, volatility: float) -> RiskMetrics:
        """
        Calculate risk metrics for Bear Call Spread.

        Args:
            strikes: [short_call_strike, long_call_strike] in cents
            underlying_price: Current underlying price in cents
            time_to_expiration: Days to expiration
            volatility: Implied volatility

        Returns:
            RiskMetrics for the Bear Call Spread
        """
        if len(strikes) != 2:
            raise ValueError("Bear Call Spread requires exactly 2 strikes")

        short_strike, long_strike = strikes

        if short_strike >= long_strike:
            raise ValueError("Short strike must be lower than long strike")

        # Calculate spread width
        spread_width = long_strike - short_strike

        # Estimate net credit (simplified)
        # Assume roughly 25-35% of spread width as net credit for OTM spreads
        otm_factor = max(0, (short_strike - underlying_price) / underlying_price)
        credit_factor = 0.3 - (otm_factor * 0.1)  # Reduce credit for deep OTM spreads
        net_credit = int(spread_width * max(credit_factor, 0.15))

        # Risk calculations
        max_profit = net_credit
        max_loss = spread_width - net_credit

        # Breakeven point
        breakeven = short_strike + net_credit
        breakevens = [breakeven]

        # Probability estimation
        # Probability price stays below breakeven
        price_move_needed = (breakeven - underlying_price) / underlying_price
        if price_move_needed <= 0:
            profit_probability = 0.4  # Price already above breakeven
        else:
            profit_probability = min(0.8, 0.7 + (price_move_needed * 0.5))

        # Risk/reward ratio
        risk_reward_ratio = max_profit / max_loss if max_loss > 0 else float('inf')

        # Capital requirements (higher for credit spreads due to margin)
        capital_requirement = max_loss
        margin_requirement = int(max_loss * 1.3)  # Higher margin for short options

        return RiskMetrics(
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=breakevens,
            profit_probability=profit_probability,
            risk_reward_ratio=risk_reward_ratio,
            capital_requirement=capital_requirement,
            margin_requirement=margin_requirement
        )

    def get_position_legs(self, strikes: List[int], expiration_date: datetime) -> List[PositionLeg]:
        """
        Get position legs for Bear Call Spread construction.

        Args:
            strikes: [short_call_strike, long_call_strike] in cents
            expiration_date: Option expiration date

        Returns:
            List of PositionLeg objects for Bear Call Spread
        """
        if len(strikes) != 2:
            raise ValueError("Bear Call Spread requires exactly 2 strikes: [short_call, long_call]")

        short_call_strike, long_call_strike = strikes

        return [
            # Short call at lower strike
            PositionLeg(
                option_type=OptionType.CALL,
                strike=short_call_strike,
                quantity=-1,  # Short position
                expiration_date=expiration_date
            ),
            # Long call at higher strike
            PositionLeg(
                option_type=OptionType.CALL,
                strike=long_call_strike,
                quantity=1,  # Long position
                expiration_date=expiration_date
            )
        ]

    def _validate_strategy_strikes(self, strikes: List[int], underlying_price: int) -> bool:
        """
        Validate Bear Call Spread strike configuration.

        Args:
            strikes: Strike prices to validate
            underlying_price: Current underlying price

        Returns:
            True if strikes are valid for Bear Call Spread
        """
        if len(strikes) != 2:
            return False

        short_strike, long_strike = strikes

        # Short strike must be lower than long strike
        if short_strike >= long_strike:
            return False

        # Spread width should be reasonable ($1 to $20)
        spread_width = long_strike - short_strike
        if spread_width < 100 or spread_width > 2000:  # $1 to $20
            return False

        # Short strike should be above current price (OTM for bearish strategy)
        if short_strike <= underlying_price:
            return False

        # Short strike should not be too far OTM (within 25% of current price)
        distance_from_price = (short_strike - underlying_price) / underlying_price
        if distance_from_price > 0.25:
            return False

        return True


class BullPutSpreadStrategy(BaseStrategy):
    """
    Bull Put Spread options strategy implementation.

    Bull Put Spread consists of:
    - Short put at higher strike (sold)
    - Long put at lower strike (purchased)

    Structure: Sell Put @ A, Buy Put @ B (A > B)

    Bullish strategy with limited upside and limited downside.
    Net credit strategy - receive premium upfront.

    Max profit: Net credit received
    Max loss: (A - B) - net credit received
    Breakeven: A - net credit received

    Best market conditions:
    - Bull trending regimes (1) and Recovery regimes (6)
    - High implied volatility that can contract
    - Expect upward or sideways price movement above short strike
    """

    def _create_metadata(self) -> StrategyMetadata:
        """Create Bull Put Spread strategy metadata."""
        return StrategyMetadata(
            name="Bull Put Spread",
            category=StrategyCategory.DIRECTIONAL,
            risk_level=RiskLevel.MEDIUM,
            capital_requirement=1.1,  # Moderate capital due to margin requirements
            description="Bullish credit spread profiting from upward/sideways moves with limited risk",
            typical_market_conditions=[
                "Bull trending market (regime 1)",
                "Recovery phase (regime 6)",
                "High implied volatility",
                "Expected upward or sideways price movement"
            ]
        )

    def get_strategy_type(self) -> StrategyType:
        """Get strategy type enum."""
        return StrategyType.BULL_PUT_SPREAD

    def validate_market_conditions(self, conditions: MarketConditions) -> bool:
        """
        Validate market conditions for Bull Put Spread strategy.

        Args:
            conditions: Current market conditions

        Returns:
            True if conditions are suitable for Bull Put Spread
        """
        # Favorable regimes: Bull trending (1) and Recovery (6)
        favorable_regimes = [1, 6]

        # Check regime suitability
        if conditions.regime not in favorable_regimes:
            return False

        # Check trend strength (should be bullish or neutral)
        if conditions.trend_strength < -0.1:  # Too bearish
            return False

        # Check volatility (higher levels work better for credit collection)
        if conditions.volatility_rank < 0.3:
            return False

        # Check time to expiration (20-60 DTE optimal)
        if conditions.time_to_expiration < 15 or conditions.time_to_expiration > 90:
            return False

        return True

    def calculate_entry_criteria(self, conditions: MarketConditions) -> EntrySignal:
        """
        Calculate entry signal for Bull Put Spread.

        Args:
            conditions: Current market conditions

        Returns:
            EntrySignal with entry decision and confidence
        """
        if not self.validate_market_conditions(conditions):
            return EntrySignal(
                should_enter=False,
                confidence=0.0,
                reasons=["Market conditions not suitable for Bull Put Spread"]
            )

        confidence = 0.0
        reasons = []

        # Regime scoring
        if conditions.regime == 1:  # Bull trending - perfect
            confidence += 0.35
            reasons.append("Bull trending regime ideal for bull put spread")
        elif conditions.regime == 6:  # Recovery - good
            confidence += 0.25
            reasons.append("Recovery regime favorable for bullish strategies")

        # Trend strength scoring (bullish is best)
        if conditions.trend_strength >= 0.4:  # Strong bullish trend
            confidence += 0.25
            reasons.append(f"Strong bullish trend {conditions.trend_strength:.2f} supports put spread")
        elif conditions.trend_strength >= 0.2:  # Moderate bullish trend
            confidence += 0.2
            reasons.append("Moderate bullish trend supports strategy")
        elif conditions.trend_strength >= -0.1:  # Neutral/weak bullish
            confidence += 0.15
            reasons.append("Neutral trend acceptable for credit spread")

        # Volatility scoring (higher is better for credit collection)
        if conditions.volatility_rank >= 0.7:  # High volatility
            confidence += 0.25
            reasons.append("High volatility excellent for credit collection")
        elif conditions.volatility_rank >= 0.5:  # Moderate-high volatility
            confidence += 0.2
            reasons.append("Good volatility level for premium collection")
        elif conditions.volatility_rank >= 0.3:  # Moderate volatility
            confidence += 0.1
            reasons.append("Moderate volatility acceptable")

        # Time to expiration scoring
        if 30 <= conditions.time_to_expiration <= 50:  # Optimal range
            confidence += 0.15
            reasons.append(f"Optimal {conditions.time_to_expiration} DTE for credit spread")
        elif 20 <= conditions.time_to_expiration < 30:  # Shorter term
            confidence += 0.1
            reasons.append("Shorter timeframe increases time decay benefit")
        elif 50 < conditions.time_to_expiration <= 90:  # Longer term
            confidence += 0.05
            reasons.append("Longer timeframe acceptable for credit collection")

        # Position sizing recommendation
        recommended_size = 1.0
        if confidence > 0.8:
            recommended_size = 1.2  # Moderate increase for high confidence
        elif confidence < 0.5:
            recommended_size = 0.8  # Conservative sizing for lower confidence

        return EntrySignal(
            should_enter=confidence >= 0.6,
            confidence=confidence,
            reasons=reasons,
            recommended_size=recommended_size
        )

    def calculate_exit_criteria(self, position: 'Position', conditions: MarketConditions) -> ExitSignal:
        """
        Calculate exit signal for existing Bull Put Spread position.

        Args:
            position: Current Bull Put Spread position
            conditions: Current market conditions

        Returns:
            ExitSignal with exit decision and reasoning
        """
        reasons = []
        urgency = 0.0
        exit_type = ""

        # Time-based exit
        days_to_exp = position.days_to_expiration
        if days_to_exp <= 7:
            urgency += 0.6
            exit_type = "time"
            reasons.append("Approaching expiration - assignment risk for short put")
        elif days_to_exp <= 15:
            urgency += 0.2
            reasons.append("Short time to expiration")

        # Profit target exit (credit spreads)
        current_pnl = position.calculate_unrealized_pnl()
        max_profit = position.calculate_max_profit()

        if max_profit > 0:
            profit_percent = current_pnl / max_profit
            if profit_percent >= 0.6:  # 60% of max profit
                urgency += 0.7
                exit_type = "profit"
                reasons.append(f"Excellent profit target reached: {profit_percent:.1%}")
            elif profit_percent >= 0.4:  # 40% of max profit
                urgency += 0.4
                exit_type = "profit"
                reasons.append(f"Good profit level: {profit_percent:.1%}")
            elif profit_percent >= 0.2:  # 20% profit consideration
                urgency += 0.2
                if not exit_type:
                    exit_type = "profit"
                reasons.append(f"Partial profit available: {profit_percent:.1%}")

        # Loss limit exit
        max_loss = position.calculate_max_loss()
        if max_loss > 0:
            loss_percent = abs(current_pnl) / max_loss
            if current_pnl < 0 and loss_percent >= 0.75:  # 75% of max loss
                urgency = 1.0
                exit_type = "loss"
                reasons.append("Approaching maximum loss threshold")
            elif current_pnl < 0 and loss_percent >= 0.5:  # 50% loss
                urgency += 0.7
                exit_type = "loss"
                reasons.append("Significant loss incurred")

        # Market regime change
        unfavorable_regimes = [2, 3, 7, 8]  # Bear trending, high vol, distribution, crisis
        if conditions.regime in unfavorable_regimes:
            urgency += 0.4
            if not exit_type:
                exit_type = "adjustment"
            reasons.append(f"Market regime {conditions.regime} unfavorable for bull strategies")

        # Trend reversal (turned bearish)
        if conditions.trend_strength < -0.3:  # Strong bearish turn
            urgency += 0.7
            if not exit_type:
                exit_type = "adjustment"
            reasons.append("Strong bearish trend threatens bull put spread")
        elif conditions.trend_strength < -0.1:  # Moderate bearish turn
            urgency += 0.4
            if not exit_type:
                exit_type = "adjustment"
            reasons.append("Bearish momentum developing")

        # Price movement below short strike (assignment risk)
        if len(position.strikes) >= 2:
            short_strike = max(position.strikes)  # Higher strike is the short put
            current_price = position.current_underlying_price

            if current_price <= short_strike:
                urgency += 0.6
                if not exit_type:
                    exit_type = "loss"
                reasons.append("Price below short put strike - high assignment risk")
            elif current_price < short_strike * 1.05:  # Within 5% of short strike
                urgency += 0.3
                reasons.append("Price approaching short put strike")

        # Volatility crush (bad for credit spreads if need to buy back)
        if conditions.volatility_rank < 0.2 and current_pnl < 0:
            urgency += 0.3
            reasons.append("Low volatility makes position adjustment expensive")

        return ExitSignal(
            should_exit=urgency >= 0.6,
            urgency=min(urgency, 1.0),
            exit_type=exit_type or "hold",
            reasons=reasons or ["Position within acceptable parameters"]
        )

    def get_risk_metrics(self, strikes: List[int], underlying_price: int,
                        time_to_expiration: int, volatility: float) -> RiskMetrics:
        """
        Calculate risk metrics for Bull Put Spread.

        Args:
            strikes: [short_put_strike, long_put_strike] in cents (short > long)
            underlying_price: Current underlying price in cents
            time_to_expiration: Days to expiration
            volatility: Implied volatility

        Returns:
            RiskMetrics for the Bull Put Spread
        """
        if len(strikes) != 2:
            raise ValueError("Bull Put Spread requires exactly 2 strikes")

        short_strike, long_strike = strikes

        if short_strike <= long_strike:
            raise ValueError("Short strike must be higher than long strike")

        # Calculate spread width
        spread_width = short_strike - long_strike

        # Estimate net credit (simplified)
        # Assume roughly 25-35% of spread width as net credit for OTM spreads
        otm_factor = max(0, (underlying_price - short_strike) / underlying_price)
        credit_factor = 0.3 + (otm_factor * 0.1)  # More credit for further OTM spreads
        net_credit = int(spread_width * min(credit_factor, 0.4))

        # Risk calculations
        max_profit = net_credit
        max_loss = spread_width - net_credit

        # Breakeven point
        breakeven = short_strike - net_credit
        breakevens = [breakeven]

        # Probability estimation
        # Probability price stays above breakeven
        price_move_needed = (underlying_price - breakeven) / underlying_price
        if price_move_needed <= 0:
            profit_probability = 0.4  # Price already below breakeven
        else:
            profit_probability = min(0.8, 0.6 + (price_move_needed * 0.5))

        # Risk/reward ratio
        risk_reward_ratio = max_profit / max_loss if max_loss > 0 else float('inf')

        # Capital requirements (higher for credit spreads due to margin)
        capital_requirement = max_loss
        margin_requirement = int(max_loss * 1.3)  # Higher margin for short options

        return RiskMetrics(
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=breakevens,
            profit_probability=profit_probability,
            risk_reward_ratio=risk_reward_ratio,
            capital_requirement=capital_requirement,
            margin_requirement=margin_requirement
        )

    def get_position_legs(self, strikes: List[int], expiration_date: datetime) -> List[PositionLeg]:
        """
        Get position legs for Bull Put Spread construction.

        Args:
            strikes: [short_put_strike, long_put_strike] in cents (short > long)
            expiration_date: Option expiration date

        Returns:
            List of PositionLeg objects for Bull Put Spread
        """
        if len(strikes) != 2:
            raise ValueError("Bull Put Spread requires exactly 2 strikes: [short_put, long_put]")

        short_put_strike, long_put_strike = strikes

        return [
            # Short put at higher strike
            PositionLeg(
                option_type=OptionType.PUT,
                strike=short_put_strike,
                quantity=-1,  # Short position
                expiration_date=expiration_date
            ),
            # Long put at lower strike
            PositionLeg(
                option_type=OptionType.PUT,
                strike=long_put_strike,
                quantity=1,  # Long position
                expiration_date=expiration_date
            )
        ]

    def _validate_strategy_strikes(self, strikes: List[int], underlying_price: int) -> bool:
        """
        Validate Bull Put Spread strike configuration.

        Args:
            strikes: Strike prices to validate
            underlying_price: Current underlying price

        Returns:
            True if strikes are valid for Bull Put Spread
        """
        if len(strikes) != 2:
            return False

        short_strike, long_strike = strikes

        # Short strike must be higher than long strike
        if short_strike <= long_strike:
            return False

        # Spread width should be reasonable ($1 to $20)
        spread_width = short_strike - long_strike
        if spread_width < 100 or spread_width > 2000:  # $1 to $20
            return False

        # Short strike should be below current price (OTM for bullish strategy)
        if short_strike >= underlying_price:
            return False

        # Short strike should not be too far OTM (within 25% below current price)
        distance_from_price = (underlying_price - short_strike) / underlying_price
        if distance_from_price > 0.25:
            return False

        return True


class BearPutSpreadStrategy(BaseStrategy):
    """
    Bear Put Spread options strategy implementation.

    Bear Put Spread consists of:
    - Long put at higher strike (purchased)
    - Short put at lower strike (sold)

    Structure: Buy Put @ A, Sell Put @ B (A > B)

    Bearish strategy with limited upside and limited downside.
    Net debit strategy - pay premium upfront.

    Max profit: (A - B) - net debit paid
    Max loss: Net debit paid
    Breakeven: A - net debit paid

    Best market conditions:
    - Bear trending regimes (2) and Distribution regimes (7)
    - Moderate volatility
    - Expect downward price movement with defined target
    """

    def _create_metadata(self) -> StrategyMetadata:
        """Create Bear Put Spread strategy metadata."""
        return StrategyMetadata(
            name="Bear Put Spread",
            category=StrategyCategory.DIRECTIONAL,
            risk_level=RiskLevel.LOW,
            capital_requirement=1.0,  # Limited risk due to debit spread nature
            description="Bearish debit spread with limited downside risk for moderate downward moves",
            typical_market_conditions=[
                "Bear trending market (regime 2)",
                "Distribution phase (regime 7)",
                "Moderate volatility environment",
                "Expected moderate downward price movement"
            ]
        )

    def get_strategy_type(self) -> StrategyType:
        """Get strategy type enum."""
        return StrategyType.BEAR_PUT_SPREAD

    def validate_market_conditions(self, conditions: MarketConditions) -> bool:
        """
        Validate market conditions for Bear Put Spread strategy.

        Args:
            conditions: Current market conditions

        Returns:
            True if conditions are suitable for Bear Put Spread
        """
        # Favorable regimes: Bear trending (2) and Distribution (7)
        favorable_regimes = [2, 7]

        # Check regime suitability
        if conditions.regime not in favorable_regimes:
            return False

        # Check trend strength (should be bearish)
        if conditions.trend_strength > -0.2:  # Need some downward bias
            return False

        # Check volatility (moderate levels work best)
        if conditions.volatility_rank < 0.2 or conditions.volatility_rank > 0.8:
            return False

        # Check time to expiration (30-60 DTE optimal)
        if conditions.time_to_expiration < 20 or conditions.time_to_expiration > 90:
            return False

        return True

    def calculate_entry_criteria(self, conditions: MarketConditions) -> EntrySignal:
        """
        Calculate entry signal for Bear Put Spread.

        Args:
            conditions: Current market conditions

        Returns:
            EntrySignal with entry decision and confidence
        """
        if not self.validate_market_conditions(conditions):
            return EntrySignal(
                should_enter=False,
                confidence=0.0,
                reasons=["Market conditions not suitable for Bear Put Spread"]
            )

        confidence = 0.0
        reasons = []

        # Regime scoring
        if conditions.regime == 2:  # Bear trending - perfect
            confidence += 0.35
            reasons.append("Bear trending regime ideal for bear put spread")
        elif conditions.regime == 7:  # Distribution - good
            confidence += 0.25
            reasons.append("Distribution regime favorable for bearish strategies")

        # Trend strength scoring
        if conditions.trend_strength <= -0.6:  # Strong bearish trend
            confidence += 0.25
            reasons.append(f"Strong bearish trend {conditions.trend_strength:.2f} supports put spread")
        elif conditions.trend_strength <= -0.4:  # Moderate bearish trend
            confidence += 0.2
            reasons.append("Moderate bearish trend supports strategy")
        elif conditions.trend_strength <= -0.2:  # Weak bearish trend
            confidence += 0.1
            reasons.append("Weak bearish bias present")

        # Volatility scoring (moderate is best)
        if 0.4 <= conditions.volatility_rank <= 0.6:  # Sweet spot
            confidence += 0.2
            reasons.append("Optimal volatility level for put spreads")
        elif 0.2 <= conditions.volatility_rank < 0.4:  # Lower but acceptable
            confidence += 0.15
            reasons.append("Lower volatility acceptable for debit spreads")
        elif 0.6 < conditions.volatility_rank <= 0.8:  # Higher but manageable
            confidence += 0.1
            reasons.append("Higher volatility increases premium cost")

        # Time to expiration scoring
        if 30 <= conditions.time_to_expiration <= 60:  # Optimal range
            confidence += 0.2
            reasons.append(f"Optimal {conditions.time_to_expiration} DTE for put spread")
        elif 20 <= conditions.time_to_expiration < 30:  # Shorter term
            confidence += 0.15
            reasons.append("Shorter time frame acceptable")
        elif 60 < conditions.time_to_expiration <= 90:  # Longer term
            confidence += 0.1
            reasons.append("Longer timeframe reduces time decay benefit")

        # Position sizing recommendation
        recommended_size = 1.0
        if confidence > 0.8:
            recommended_size = 1.25  # Increase size for high confidence
        elif confidence < 0.5:
            recommended_size = 0.75  # Reduce size for lower confidence

        return EntrySignal(
            should_enter=confidence >= 0.6,
            confidence=confidence,
            reasons=reasons,
            recommended_size=recommended_size
        )

    def calculate_exit_criteria(self, position: 'Position', conditions: MarketConditions) -> ExitSignal:
        """
        Calculate exit signal for existing Bear Put Spread position.

        Args:
            position: Current Bear Put Spread position
            conditions: Current market conditions

        Returns:
            ExitSignal with exit decision and reasoning
        """
        reasons = []
        urgency = 0.0
        exit_type = ""

        # Time-based exit
        days_to_exp = position.days_to_expiration
        if days_to_exp <= 5:
            urgency += 0.7
            exit_type = "time"
            reasons.append("Approaching expiration - high gamma risk")
        elif days_to_exp <= 10:
            urgency += 0.3
            reasons.append("Short time to expiration")

        # Profit target exit
        current_pnl = position.calculate_unrealized_pnl()
        max_profit = position.calculate_max_profit()

        if max_profit > 0:
            profit_percent = current_pnl / max_profit
            if profit_percent >= 0.75:  # 75% of max profit
                urgency += 0.8
                exit_type = "profit"
                reasons.append(f"Excellent profit target reached: {profit_percent:.1%}")
            elif profit_percent >= 0.5:  # 50% of max profit
                urgency += 0.5
                exit_type = "profit"
                reasons.append(f"Good profit level: {profit_percent:.1%}")
            elif profit_percent >= 0.25:  # 25% profit consideration
                urgency += 0.2
                if not exit_type:
                    exit_type = "profit"
                reasons.append(f"Partial profit available: {profit_percent:.1%}")

        # Loss limit exit
        max_loss = position.calculate_max_loss()
        if max_loss > 0:
            loss_percent = abs(current_pnl) / max_loss
            if current_pnl < 0 and loss_percent >= 0.8:  # 80% of max loss
                urgency = 1.0
                exit_type = "loss"
                reasons.append("Approaching maximum loss threshold")
            elif current_pnl < 0 and loss_percent >= 0.5:  # 50% loss
                urgency += 0.6
                exit_type = "loss"
                reasons.append("Significant loss incurred")

        # Market regime change
        unfavorable_regimes = [1, 3, 6, 8]  # Bull trending, high vol, recovery, crisis
        if conditions.regime in unfavorable_regimes:
            urgency += 0.4
            if not exit_type:
                exit_type = "adjustment"
            reasons.append(f"Market regime {conditions.regime} unfavorable for bear strategies")

        # Trend reversal
        if conditions.trend_strength > 0.2:  # Turned bullish
            urgency += 0.6
            if not exit_type:
                exit_type = "adjustment"
            reasons.append("Trend has turned bullish")
        elif conditions.trend_strength > -0.1:  # Lost bearish bias
            urgency += 0.3
            if not exit_type:
                exit_type = "adjustment"
            reasons.append("Bearish momentum weakening")

        # Volatility expansion (can help if in-the-money)
        if conditions.volatility_rank > 0.9:
            if current_pnl > 0:
                urgency += 0.2
                reasons.append("High volatility - consider profit taking")
            else:
                urgency += 0.4
                reasons.append("High volatility may increase losses")

        # Price target reached (near lower strike)
        if len(position.strikes) >= 2:
            long_strike = max(position.strikes)  # Higher strike is the long put
            short_strike = min(position.strikes)  # Lower strike is the short put
            current_price = position.current_underlying_price

            # Check if price is near short strike (profit zone)
            distance_to_short = abs(current_price - short_strike) / short_strike
            if distance_to_short < 0.05:  # Within 5% of short strike
                urgency += 0.6
                exit_type = "profit"
                reasons.append("Price approaching maximum profit zone")

            # Check if price rose above long strike
            if current_price > long_strike:
                urgency += 0.5
                if not exit_type:
                    exit_type = "loss"
                reasons.append("Price above long put strike")

        return ExitSignal(
            should_exit=urgency >= 0.6,
            urgency=min(urgency, 1.0),
            exit_type=exit_type or "hold",
            reasons=reasons or ["Position within acceptable parameters"]
        )

    def get_risk_metrics(self, strikes: List[int], underlying_price: int,
                        time_to_expiration: int, volatility: float) -> RiskMetrics:
        """
        Calculate risk metrics for Bear Put Spread.

        Args:
            strikes: [long_put_strike, short_put_strike] in cents (long > short)
            underlying_price: Current underlying price in cents
            time_to_expiration: Days to expiration
            volatility: Implied volatility

        Returns:
            RiskMetrics for the Bear Put Spread
        """
        if len(strikes) != 2:
            raise ValueError("Bear Put Spread requires exactly 2 strikes")

        long_strike, short_strike = strikes

        if long_strike <= short_strike:
            raise ValueError("Long strike must be higher than short strike")

        # Calculate spread width
        spread_width = long_strike - short_strike

        # Estimate net debit (simplified)
        # Assume roughly 30-40% of spread width as net debit for ATM spreads
        atm_factor = abs(underlying_price - long_strike) / underlying_price
        debit_factor = 0.35 - (atm_factor * 0.1)  # Reduce debit for OTM spreads
        net_debit = int(spread_width * max(debit_factor, 0.15))

        # Risk calculations
        max_profit = spread_width - net_debit
        max_loss = net_debit

        # Breakeven point
        breakeven = long_strike - net_debit
        breakevens = [breakeven]

        # Probability estimation
        # Simple model: probability price ends below breakeven
        price_move_needed = (underlying_price - breakeven) / underlying_price
        # Rough approximation based on normal distribution
        if price_move_needed >= 0:
            profit_probability = 0.7  # Already below breakeven
        else:
            profit_probability = max(0.3, 0.6 + (price_move_needed * 2))

        # Risk/reward ratio
        risk_reward_ratio = max_profit / max_loss if max_loss > 0 else float('inf')

        # Capital requirements
        capital_requirement = max_loss  # Net debit paid
        margin_requirement = int(max_loss * 1.1)  # Small buffer for debit spreads

        return RiskMetrics(
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=breakevens,
            profit_probability=profit_probability,
            risk_reward_ratio=risk_reward_ratio,
            capital_requirement=capital_requirement,
            margin_requirement=margin_requirement
        )

    def get_position_legs(self, strikes: List[int], expiration_date: datetime) -> List[PositionLeg]:
        """
        Get position legs for Bear Put Spread construction.

        Args:
            strikes: [long_put_strike, short_put_strike] in cents (long > short)
            expiration_date: Option expiration date

        Returns:
            List of PositionLeg objects for Bear Put Spread
        """
        if len(strikes) != 2:
            raise ValueError("Bear Put Spread requires exactly 2 strikes: [long_put, short_put]")

        long_put_strike, short_put_strike = strikes

        return [
            # Long put at higher strike
            PositionLeg(
                option_type=OptionType.PUT,
                strike=long_put_strike,
                quantity=1,  # Long position
                expiration_date=expiration_date
            ),
            # Short put at lower strike
            PositionLeg(
                option_type=OptionType.PUT,
                strike=short_put_strike,
                quantity=-1,  # Short position
                expiration_date=expiration_date
            )
        ]

    def _validate_strategy_strikes(self, strikes: List[int], underlying_price: int) -> bool:
        """
        Validate Bear Put Spread strike configuration.

        Args:
            strikes: Strike prices to validate
            underlying_price: Current underlying price

        Returns:
            True if strikes are valid for Bear Put Spread
        """
        if len(strikes) != 2:
            return False

        long_strike, short_strike = strikes

        # Long strike must be higher than short strike
        if long_strike <= short_strike:
            return False

        # Spread width should be reasonable ($1 to $20)
        spread_width = long_strike - short_strike
        if spread_width < 100 or spread_width > 2000:  # $1 to $20
            return False

        # Long strike should be reasonably close to current price (within 20%)
        distance_from_price = abs(long_strike - underlying_price) / underlying_price
        if distance_from_price > 0.20:
            return False

        # Short strike should be below current price for bearish strategy
        if short_strike >= underlying_price:
            return False

        return True

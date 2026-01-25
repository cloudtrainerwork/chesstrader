"""
Neutral options strategies implementation.

Implements Iron Condor and Iron Butterfly strategies for neutral market outlook.
These strategies profit from low volatility and sideways price movement.
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


class IronCondorStrategy(BaseStrategy):
    """
    Iron Condor options strategy implementation.

    Iron Condor consists of:
    - Short put spread (sell higher strike put, buy lower strike put)
    - Short call spread (sell lower strike call, buy higher strike call)

    Structure: Buy Put @ A, Sell Put @ B, Sell Call @ C, Buy Call @ D
    Where: A < B < C < D (all strikes)

    Profit zone: Between B and C (short strikes)
    Max profit: Net credit received
    Max loss: (Spread width - Net credit) per spread

    Best market conditions:
    - Low volatility regimes (regimes 2, 4)
    - Sideways market movement
    - High implied volatility that can contract
    """

    def _create_metadata(self) -> StrategyMetadata:
        """Create Iron Condor strategy metadata."""
        return StrategyMetadata(
            name="Iron Condor",
            category=StrategyCategory.NEUTRAL,
            risk_level=RiskLevel.MEDIUM,
            capital_requirement=1.5,  # Moderate capital requirement
            description="Credit spread combining short put and call spreads for range-bound profit",
            typical_market_conditions=[
                "Low volatility environment",
                "Sideways price action expected",
                "High implied volatility",
                "Stable market regime"
            ]
        )

    def get_strategy_type(self) -> StrategyType:
        """Get strategy type enum."""
        return StrategyType.IRON_CONDOR

    def validate_market_conditions(self, conditions: MarketConditions) -> bool:
        """
        Validate market conditions for Iron Condor strategy.

        Args:
            conditions: Current market conditions

        Returns:
            True if conditions are suitable for Iron Condor
        """
        # Favorable regimes: low volatility regimes (2, 4)
        favorable_regimes = [2, 4]

        # Check regime suitability
        if conditions.regime not in favorable_regimes:
            return False

        # Check volatility conditions
        if conditions.volatility_rank < 0.3:  # Need some volatility to collect premium
            return False

        # Check time to expiration (works best 30-45 DTE)
        if conditions.time_to_expiration < 20 or conditions.time_to_expiration > 60:
            return False

        # Check trend strength (prefer weak trends)
        if abs(conditions.trend_strength) > 0.6:  # Too trending
            return False

        return True

    def calculate_entry_criteria(self, conditions: MarketConditions) -> EntrySignal:
        """
        Calculate entry signal for Iron Condor.

        Args:
            conditions: Current market conditions

        Returns:
            EntrySignal with entry decision and confidence
        """
        if not self.validate_market_conditions(conditions):
            return EntrySignal(
                should_enter=False,
                confidence=0.0,
                reasons=["Market conditions not suitable for Iron Condor"]
            )

        confidence = 0.0
        reasons = []

        # Regime scoring
        if conditions.regime in [2, 4]:  # Perfect regimes
            confidence += 0.3
            reasons.append(f"Favorable regime {conditions.regime} for neutral strategy")

        # Volatility scoring
        if 0.4 <= conditions.volatility_rank <= 0.8:  # Good volatility range
            confidence += 0.25
            reasons.append(f"Good volatility rank {conditions.volatility_rank:.2f} for credit collection")
        elif 0.3 <= conditions.volatility_rank < 0.4:  # Acceptable
            confidence += 0.15
            reasons.append("Acceptable volatility for premium collection")

        # Time to expiration scoring
        if 30 <= conditions.time_to_expiration <= 45:  # Optimal range
            confidence += 0.2
            reasons.append(f"Optimal {conditions.time_to_expiration} DTE for Iron Condor")
        elif 20 <= conditions.time_to_expiration < 30:  # Acceptable
            confidence += 0.1
            reasons.append("Acceptable time to expiration")

        # Trend strength scoring (prefer sideways)
        if abs(conditions.trend_strength) < 0.3:  # Sideways
            confidence += 0.25
            reasons.append("Weak trend favorable for neutral strategy")
        elif abs(conditions.trend_strength) < 0.6:  # Moderate trend
            confidence += 0.1
            reasons.append("Moderate trend acceptable for wide strikes")

        # Position sizing recommendation
        recommended_size = 1.0
        if confidence > 0.7:
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
        Calculate exit signal for existing Iron Condor position.

        Args:
            position: Current Iron Condor position
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
            urgency += 0.5
            exit_type = "time"
            reasons.append("Approaching expiration - high gamma risk")

        # Profit target exit
        current_pnl = position.calculate_unrealized_pnl()
        max_profit = position.calculate_max_profit()

        if max_profit > 0:
            profit_percent = current_pnl / max_profit
            if profit_percent >= 0.5:  # 50% profit target
                urgency += 0.7
                exit_type = "profit"
                reasons.append(f"Profit target reached: {profit_percent:.1%} of max profit")
            elif profit_percent >= 0.25:  # 25% profit consideration
                urgency += 0.3
                exit_type = "profit"
                reasons.append(f"Partial profit available: {profit_percent:.1%} of max profit")

        # Loss limit exit
        max_loss = position.calculate_max_loss()
        if max_loss > 0:
            loss_percent = abs(current_pnl) / max_loss
            if current_pnl < 0 and loss_percent >= 2.0:  # 200% of premium lost
                urgency = 1.0
                exit_type = "loss"
                reasons.append("Maximum loss threshold breached")
            elif current_pnl < 0 and loss_percent >= 1.5:  # 150% of premium
                urgency += 0.8
                exit_type = "loss"
                reasons.append("Approaching maximum loss threshold")

        # Volatility expansion exit
        if conditions.volatility_rank > 0.9:  # Very high volatility
            urgency += 0.4
            if not exit_type:
                exit_type = "adjustment"
            reasons.append("High volatility may require position adjustment")

        # Market regime change
        unfavorable_regimes = [0, 1, 5, 6, 7]  # High volatility/trending regimes
        if conditions.regime in unfavorable_regimes:
            urgency += 0.3
            if not exit_type:
                exit_type = "adjustment"
            reasons.append(f"Market regime {conditions.regime} unfavorable for Iron Condor")

        # Breakeven test
        breakevens = position.calculate_breakevens()
        current_price = position.current_underlying_price

        if breakevens and len(breakevens) >= 2:
            lower_breakeven = min(breakevens)
            upper_breakeven = max(breakevens)

            # Check if price is approaching breakevens
            distance_to_lower = abs(current_price - lower_breakeven) / current_price
            distance_to_upper = abs(current_price - upper_breakeven) / current_price

            if distance_to_lower < 0.02 or distance_to_upper < 0.02:  # Within 2%
                urgency += 0.6
                if not exit_type:
                    exit_type = "adjustment"
                reasons.append("Price approaching breakeven levels")

        return ExitSignal(
            should_exit=urgency >= 0.6,
            urgency=min(urgency, 1.0),
            exit_type=exit_type or "hold",
            reasons=reasons or ["Position within acceptable parameters"]
        )

    def get_risk_metrics(self, strikes: List[int], underlying_price: int,
                        time_to_expiration: int, volatility: float) -> RiskMetrics:
        """
        Calculate risk metrics for Iron Condor.

        Args:
            strikes: [short_put, long_put, long_call, short_call] strikes in cents
            underlying_price: Current underlying price in cents
            time_to_expiration: Days to expiration
            volatility: Implied volatility

        Returns:
            RiskMetrics for the Iron Condor
        """
        if len(strikes) != 4:
            raise ValueError("Iron Condor requires exactly 4 strikes")

        short_put, long_put, long_call, short_call = strikes

        # Calculate spread widths
        put_spread_width = short_put - long_put
        call_spread_width = short_call - long_call

        # Estimate net credit (simplified - would use actual option prices in production)
        # Assume roughly 1/3 of spread width as credit for each spread
        put_credit = put_spread_width * 0.33
        call_credit = call_spread_width * 0.33
        net_credit = put_credit + call_credit

        # Risk calculations
        max_profit = int(net_credit)
        max_loss_per_spread = max(put_spread_width, call_spread_width) - int(net_credit)
        max_loss = int(max_loss_per_spread)

        # Breakeven points
        breakeven_lower = short_put - int(net_credit)
        breakeven_upper = long_call + int(net_credit)
        breakevens = [breakeven_lower, breakeven_upper]

        # Probability estimation (simplified)
        profit_zone_width = long_call - short_put
        total_range = underlying_price * 0.4  # Assume 40% range consideration
        profit_probability = min(profit_zone_width / total_range, 0.85)

        # Risk/reward ratio
        risk_reward_ratio = max_profit / max_loss if max_loss > 0 else float('inf')

        # Capital requirements
        capital_requirement = max_loss  # Need to cover potential loss
        margin_requirement = int(max_loss * 1.2)  # 20% buffer for margin

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
        Get position legs for Iron Condor construction.

        Args:
            strikes: [short_put, long_put, long_call, short_call] strikes in cents
            expiration_date: Option expiration date

        Returns:
            List of PositionLeg objects for Iron Condor
        """
        if len(strikes) != 4:
            raise ValueError("Iron Condor requires exactly 4 strikes: [short_put, long_put, long_call, short_call]")

        short_put_strike, long_put_strike, long_call_strike, short_call_strike = strikes

        return [
            # Put spread: Long the lower strike, short the higher strike
            PositionLeg(
                option_type=OptionType.PUT,
                strike=long_put_strike,
                quantity=1,  # Long position
                expiration_date=expiration_date
            ),
            PositionLeg(
                option_type=OptionType.PUT,
                strike=short_put_strike,
                quantity=-1,  # Short position
                expiration_date=expiration_date
            ),
            # Call spread: Short the lower strike, long the higher strike
            PositionLeg(
                option_type=OptionType.CALL,
                strike=long_call_strike,
                quantity=-1,  # Short position
                expiration_date=expiration_date
            ),
            PositionLeg(
                option_type=OptionType.CALL,
                strike=short_call_strike,
                quantity=1,  # Long position
                expiration_date=expiration_date
            )
        ]

    def _validate_strategy_strikes(self, strikes: List[int], underlying_price: int) -> bool:
        """
        Validate Iron Condor strike configuration.

        Args:
            strikes: Strike prices to validate
            underlying_price: Current underlying price

        Returns:
            True if strikes are valid for Iron Condor
        """
        if len(strikes) != 4:
            return False

        short_put, long_put, long_call, short_call = strikes

        # Check strike ordering: long_put < short_put < long_call < short_call
        if not (long_put < short_put < long_call < short_call):
            return False

        # Check that short strikes are around the current price
        if short_put > underlying_price or long_call < underlying_price:
            return False

        # Check spread widths are reasonable (between $1 and $10)
        put_spread_width = short_put - long_put
        call_spread_width = short_call - long_call

        if put_spread_width < 100 or put_spread_width > 1000:  # $1 to $10
            return False

        if call_spread_width < 100 or call_spread_width > 1000:  # $1 to $10
            return False

        # Check that spreads are reasonably balanced
        width_ratio = call_spread_width / put_spread_width
        if width_ratio < 0.5 or width_ratio > 2.0:  # Within 2:1 ratio
            return False

        return True


class IronButterflyStrategy(BaseStrategy):
    """
    Iron Butterfly options strategy implementation.

    Iron Butterfly consists of:
    - Put spread and call spread with shared middle strike
    - Structure: Buy Put @ A, Sell Put @ B, Sell Call @ B, Buy Call @ C
    - Where: A < B < C (B is the body/middle strike)

    Tighter profit zone than Iron Condor but higher premium collection.

    Profit zone: Very narrow around middle strike B
    Max profit: Net credit received
    Max loss: (Spread width - Net credit)

    Best market conditions:
    - Very low volatility regimes (regime 4)
    - High confidence in sideways movement
    - Pinning expected near middle strike
    """

    def _create_metadata(self) -> StrategyMetadata:
        """Create Iron Butterfly strategy metadata."""
        return StrategyMetadata(
            name="Iron Butterfly",
            category=StrategyCategory.NEUTRAL,
            risk_level=RiskLevel.MEDIUM,
            capital_requirement=1.3,  # Moderate capital requirement
            description="Credit spread with tight profit zone around center strike for precise neutral plays",
            typical_market_conditions=[
                "Very low volatility environment",
                "High confidence in price pinning",
                "Elevated implied volatility",
                "Stable regime with minimal movement expected"
            ]
        )

    def get_strategy_type(self) -> StrategyType:
        """Get strategy type enum."""
        return StrategyType.BUTTERFLY

    def validate_market_conditions(self, conditions: MarketConditions) -> bool:
        """
        Validate market conditions for Iron Butterfly strategy.

        Args:
            conditions: Current market conditions

        Returns:
            True if conditions are suitable for Iron Butterfly
        """
        # Favorable regime: very low volatility (regime 4)
        if conditions.regime != 4:
            return False

        # Need higher volatility than Iron Condor to justify tighter strikes
        if conditions.volatility_rank < 0.4:
            return False

        # Check time to expiration (works best 20-40 DTE)
        if conditions.time_to_expiration < 15 or conditions.time_to_expiration > 50:
            return False

        # Check trend strength (prefer very weak trends)
        if abs(conditions.trend_strength) > 0.4:  # Even tighter than Iron Condor
            return False

        return True

    def calculate_entry_criteria(self, conditions: MarketConditions) -> EntrySignal:
        """
        Calculate entry signal for Iron Butterfly.

        Args:
            conditions: Current market conditions

        Returns:
            EntrySignal with entry decision and confidence
        """
        if not self.validate_market_conditions(conditions):
            return EntrySignal(
                should_enter=False,
                confidence=0.0,
                reasons=["Market conditions not suitable for Iron Butterfly"]
            )

        confidence = 0.0
        reasons = []

        # Regime scoring (very strict)
        if conditions.regime == 4:  # Perfect regime
            confidence += 0.35
            reasons.append("Optimal regime 4 for neutral butterfly strategy")

        # Volatility scoring (need higher IV than Iron Condor)
        if 0.6 <= conditions.volatility_rank <= 0.9:  # High volatility range
            confidence += 0.3
            reasons.append(f"High volatility rank {conditions.volatility_rank:.2f} suitable for tight strikes")
        elif 0.4 <= conditions.volatility_rank < 0.6:  # Moderate
            confidence += 0.2
            reasons.append("Moderate volatility acceptable for butterfly")

        # Time to expiration scoring
        if 25 <= conditions.time_to_expiration <= 35:  # Optimal range
            confidence += 0.2
            reasons.append(f"Optimal {conditions.time_to_expiration} DTE for Iron Butterfly")
        elif 15 <= conditions.time_to_expiration < 25:  # Acceptable
            confidence += 0.1
            reasons.append("Short-term expiration acceptable")

        # Trend strength scoring (very strict)
        if abs(conditions.trend_strength) < 0.2:  # Very sideways
            confidence += 0.15
            reasons.append("Very weak trend ideal for butterfly")
        elif abs(conditions.trend_strength) < 0.4:  # Weak trend
            confidence += 0.05
            reasons.append("Weak trend acceptable for butterfly")

        # Position sizing (more conservative than Iron Condor)
        recommended_size = 0.8  # Start conservative
        if confidence > 0.8:
            recommended_size = 1.0
        elif confidence < 0.6:
            recommended_size = 0.5

        return EntrySignal(
            should_enter=confidence >= 0.7,  # Higher threshold than Iron Condor
            confidence=confidence,
            reasons=reasons,
            recommended_size=recommended_size
        )

    def calculate_exit_criteria(self, position: 'Position', conditions: MarketConditions) -> ExitSignal:
        """
        Calculate exit signal for existing Iron Butterfly position.

        Args:
            position: Current Iron Butterfly position
            conditions: Current market conditions

        Returns:
            ExitSignal with exit decision and reasoning
        """
        reasons = []
        urgency = 0.0
        exit_type = ""

        # Time-based exit (more aggressive than Iron Condor)
        days_to_exp = position.days_to_expiration
        if days_to_exp <= 5:  # Earlier than Iron Condor
            urgency += 0.7
            exit_type = "time"
            reasons.append("Approaching expiration - very high gamma risk for butterfly")

        # Profit target exit (more aggressive)
        current_pnl = position.calculate_unrealized_pnl()
        max_profit = position.calculate_max_profit()

        if max_profit > 0:
            profit_percent = current_pnl / max_profit
            if profit_percent >= 0.4:  # 40% profit target (vs 50% for condor)
                urgency += 0.8
                exit_type = "profit"
                reasons.append(f"Profit target reached: {profit_percent:.1%} of max profit")
            elif profit_percent >= 0.2:  # 20% profit consideration
                urgency += 0.4
                exit_type = "profit"
                reasons.append(f"Partial profit available: {profit_percent:.1%} of max profit")

        # Loss limit exit
        max_loss = position.calculate_max_loss()
        if max_loss > 0:
            loss_percent = abs(current_pnl) / max_loss
            if current_pnl < 0 and loss_percent >= 1.5:  # More aggressive than condor
                urgency = 1.0
                exit_type = "loss"
                reasons.append("Maximum loss threshold breached")
            elif current_pnl < 0 and loss_percent >= 1.0:  # At break-even
                urgency += 0.6
                exit_type = "loss"
                reasons.append("Approaching loss threshold")

        # Volatility expansion exit (more sensitive)
        if conditions.volatility_rank > 0.8:  # Lower threshold than condor
            urgency += 0.5
            if not exit_type:
                exit_type = "adjustment"
            reasons.append("Volatility expansion threatening tight butterfly structure")

        # Market regime change (very sensitive)
        if conditions.regime != 4:
            urgency += 0.4
            if not exit_type:
                exit_type = "adjustment"
            reasons.append(f"Market regime {conditions.regime} no longer optimal for butterfly")

        # Price movement away from center (very sensitive)
        strikes = [leg.strike for leg in self.get_position_legs([100, 200, 300], datetime.now())]
        if strikes:
            center_strike = strikes[1] if len(strikes) > 1 else strikes[0]  # Middle strike
            current_price = position.current_underlying_price

            distance_from_center = abs(current_price - center_strike) / center_strike
            if distance_from_center > 0.03:  # 3% away from center
                urgency += 0.6
                if not exit_type:
                    exit_type = "adjustment"
                reasons.append("Price moving away from butterfly center strike")

        return ExitSignal(
            should_exit=urgency >= 0.5,  # More aggressive than condor
            urgency=min(urgency, 1.0),
            exit_type=exit_type or "hold",
            reasons=reasons or ["Position within acceptable parameters"]
        )

    def get_risk_metrics(self, strikes: List[int], underlying_price: int,
                        time_to_expiration: int, volatility: float) -> RiskMetrics:
        """
        Calculate risk metrics for Iron Butterfly.

        Args:
            strikes: [long_put, body_strike, long_call] strikes in cents
            underlying_price: Current underlying price in cents
            time_to_expiration: Days to expiration
            volatility: Implied volatility

        Returns:
            RiskMetrics for the Iron Butterfly
        """
        if len(strikes) != 3:
            raise ValueError("Iron Butterfly requires exactly 3 strikes")

        long_put_strike, body_strike, long_call_strike = strikes

        # Calculate spread widths (should be equal for standard butterfly)
        put_spread_width = body_strike - long_put_strike
        call_spread_width = long_call_strike - body_strike

        # Estimate net credit (higher than condor due to tighter strikes)
        # Butterfly typically collects more premium due to selling ATM options
        put_credit = put_spread_width * 0.4
        call_credit = call_spread_width * 0.4
        net_credit = put_credit + call_credit

        # Risk calculations
        max_profit = int(net_credit)
        max_loss_per_spread = max(put_spread_width, call_spread_width) - int(net_credit)
        max_loss = int(max_loss_per_spread)

        # Breakeven points (much tighter than condor)
        breakeven_lower = body_strike - int(net_credit)
        breakeven_upper = body_strike + int(net_credit)
        breakevens = [breakeven_lower, breakeven_upper]

        # Probability estimation (lower than condor due to tight range)
        profit_zone_width = breakeven_upper - breakeven_lower
        total_range = underlying_price * 0.4
        profit_probability = min(profit_zone_width / total_range, 0.6)  # Lower than condor

        # Risk/reward ratio
        risk_reward_ratio = max_profit / max_loss if max_loss > 0 else float('inf')

        # Capital requirements
        capital_requirement = max_loss
        margin_requirement = int(max_loss * 1.25)  # Slightly higher margin buffer

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
        Get position legs for Iron Butterfly construction.

        Args:
            strikes: [long_put, body_strike, long_call] strikes in cents
            expiration_date: Option expiration date

        Returns:
            List of PositionLeg objects for Iron Butterfly
        """
        if len(strikes) != 3:
            raise ValueError("Iron Butterfly requires exactly 3 strikes: [long_put, body_strike, long_call]")

        long_put_strike, body_strike, long_call_strike = strikes

        return [
            # Long put at lower strike
            PositionLeg(
                option_type=OptionType.PUT,
                strike=long_put_strike,
                quantity=1,  # Long position
                expiration_date=expiration_date
            ),
            # Short put at body strike
            PositionLeg(
                option_type=OptionType.PUT,
                strike=body_strike,
                quantity=-1,  # Short position
                expiration_date=expiration_date
            ),
            # Short call at body strike
            PositionLeg(
                option_type=OptionType.CALL,
                strike=body_strike,
                quantity=-1,  # Short position
                expiration_date=expiration_date
            ),
            # Long call at upper strike
            PositionLeg(
                option_type=OptionType.CALL,
                strike=long_call_strike,
                quantity=1,  # Long position
                expiration_date=expiration_date
            )
        ]

    def _validate_strategy_strikes(self, strikes: List[int], underlying_price: int) -> bool:
        """
        Validate Iron Butterfly strike configuration.

        Args:
            strikes: Strike prices to validate
            underlying_price: Current underlying price

        Returns:
            True if strikes are valid for Iron Butterfly
        """
        if len(strikes) != 3:
            return False

        long_put_strike, body_strike, long_call_strike = strikes

        # Check strike ordering: long_put < body < long_call
        if not (long_put_strike < body_strike < long_call_strike):
            return False

        # Body strike should be close to current price (ATM or near ATM)
        distance_from_atm = abs(body_strike - underlying_price) / underlying_price
        if distance_from_atm > 0.05:  # Within 5% of ATM
            return False

        # Check spread widths are equal (standard butterfly)
        put_spread_width = body_strike - long_put_strike
        call_spread_width = long_call_strike - body_strike

        if abs(put_spread_width - call_spread_width) > 50:  # Allow $0.50 difference
            return False

        # Check spread widths are reasonable ($2 to $20)
        if put_spread_width < 200 or put_spread_width > 2000:
            return False

        return True
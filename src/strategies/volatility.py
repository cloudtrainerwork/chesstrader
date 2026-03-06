"""
Volatility-based options strategies implementation.

Implements Straddle and Strangle strategies for high/low volatility market conditions.
These strategies profit from volatility changes regardless of price direction.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from .base import (
    BaseStrategy, StrategyMetadata, StrategyCategory, RiskLevel,
    MarketConditions, EntrySignal, ExitSignal, RiskMetrics, PositionLeg,
    OptionType, StrategyType
)

if TYPE_CHECKING:
    from ..features.position_models import Position


class VolatilityAnalysis:
    """
    Advanced volatility analysis tools for volatility strategies.

    Provides methods for volatility ranking, percentile analysis, vega exposure
    management, and regime transition detection specifically for volatility-based
    options strategies.
    """

    @staticmethod
    def calculate_volatility_percentile(current_vol: float, vol_history: List[float],
                                      lookback_days: int = 252) -> float:
        """
        Calculate volatility percentile ranking.

        Args:
            current_vol: Current implied volatility
            vol_history: Historical volatility data
            lookback_days: Number of days to look back for percentile calculation

        Returns:
            Percentile rank (0.0 to 1.0) where 1.0 is highest volatility
        """
        if not vol_history:
            return 0.5  # Neutral if no history

        recent_history = vol_history[-lookback_days:] if len(vol_history) > lookback_days else vol_history

        if len(recent_history) < 10:  # Need minimum data points
            return 0.5

        below_current = sum(1 for vol in recent_history if vol < current_vol)
        percentile = below_current / len(recent_history)

        return max(0.0, min(1.0, percentile))

    @staticmethod
    def detect_volatility_regime_transition(vol_rank_history: List[float],
                                          current_regime: int) -> Tuple[bool, int]:
        """
        Detect potential volatility regime transitions.

        Args:
            vol_rank_history: Recent volatility rank history
            current_regime: Current market regime (0-7)

        Returns:
            Tuple of (transition_detected, likely_new_regime)
        """
        if len(vol_rank_history) < 5:
            return False, current_regime

        recent_values = vol_rank_history[-5:]
        recent_avg = sum(recent_values) / len(recent_values)
        latest_value = recent_values[-1]

        # Detect transitions based on volatility patterns
        if current_regime == 4 and (recent_avg >= 0.5 or latest_value >= 0.6):  # Low vol → High vol
            return True, 3
        elif current_regime == 3 and (recent_avg <= 0.5 or latest_value <= 0.4):  # High vol → Low vol
            return True, 4
        elif recent_avg > 0.9:  # Extreme volatility conditions
            return True, 8  # Crisis regime

        return False, current_regime

    @staticmethod
    def calculate_vega_exposure_limit(portfolio_size: int, volatility_rank: float,
                                    max_vega_pct: float = 0.05) -> int:
        """
        Calculate maximum recommended vega exposure.

        Args:
            portfolio_size: Total portfolio size in cents
            volatility_rank: Current volatility percentile
            max_vega_pct: Maximum vega as percentage of portfolio

        Returns:
            Maximum vega exposure in cents
        """
        # Reduce vega exposure in high volatility environments
        vol_adjustment = 1.0 - (volatility_rank * 0.3)  # Reduce up to 30% in high vol

        max_vega = int(portfolio_size * max_vega_pct * vol_adjustment)
        return max(max_vega, portfolio_size // 1000)  # Minimum exposure

    @staticmethod
    def estimate_gamma_risk(strikes: List[int], underlying_price: int,
                          time_to_expiration: int, volatility: float) -> Dict[str, float]:
        """
        Estimate gamma risk for volatility positions.

        Args:
            strikes: Strike prices involved
            underlying_price: Current underlying price
            time_to_expiration: Days to expiration
            volatility: Current implied volatility

        Returns:
            Dictionary with gamma risk metrics
        """
        # Simplified gamma estimation (would use actual Greeks in production)
        days_factor = max(1, time_to_expiration)
        vol_factor = volatility

        # ATM options have highest gamma
        gamma_scores = []
        for strike in strikes:
            distance_from_atm = abs(strike - underlying_price) / underlying_price
            atm_gamma = (1.0 - distance_from_atm * 2) * vol_factor / days_factor
            gamma_scores.append(max(0.0, atm_gamma))

        total_gamma_risk = sum(gamma_scores)
        max_gamma_risk = max(gamma_scores) if gamma_scores else 0.0

        return {
            "total_gamma_risk": total_gamma_risk,
            "max_gamma_risk": max_gamma_risk,
            "gamma_acceleration": total_gamma_risk / max(days_factor / 30, 0.1),
            "days_to_max_gamma": days_factor
        }

    @staticmethod
    def forecast_volatility_mean_reversion(current_vol_rank: float,
                                         vol_history: List[float]) -> Dict[str, float]:
        """
        Forecast volatility mean reversion probability.

        Args:
            current_vol_rank: Current volatility percentile
            vol_history: Historical volatility data

        Returns:
            Dictionary with mean reversion forecasts
        """
        if len(vol_history) < 20:
            return {"mean_reversion_prob": 0.5, "expected_direction": 0.0, "confidence": 0.0}

        # Calculate historical mean
        mean_vol_rank = sum(vol_history[-60:]) / min(len(vol_history), 60) if vol_history else 0.5

        # Mean reversion tendency
        distance_from_mean = current_vol_rank - mean_vol_rank

        # Higher probability of reversion when further from mean
        reversion_prob = min(0.8, 0.5 + abs(distance_from_mean) * 0.6)

        # Expected direction (toward mean)
        expected_direction = -distance_from_mean

        # Confidence based on historical consistency
        vol_stability = 1.0 - (max(vol_history[-20:]) - min(vol_history[-20:])) if len(vol_history) >= 20 else 0.5
        confidence = vol_stability * 0.8

        return {
            "mean_reversion_prob": reversion_prob,
            "expected_direction": expected_direction,
            "confidence": confidence,
            "mean_vol_rank": mean_vol_rank
        }

    @staticmethod
    def calculate_volatility_skew_impact(put_vol: float, call_vol: float,
                                       underlying_price: int, strikes: List[int]) -> Dict[str, float]:
        """
        Calculate impact of volatility skew on strategy pricing.

        Args:
            put_vol: Implied volatility for puts
            call_vol: Implied volatility for calls
            underlying_price: Current underlying price
            strikes: Strategy strikes

        Returns:
            Dictionary with skew impact analysis
        """
        vol_skew = round(put_vol - call_vol, 2)

        # Analyze skew impact on each strike
        skew_impacts = []
        for strike in strikes:
            if strike < underlying_price:  # Put side
                skew_impact = vol_skew * 0.5  # Positive skew favors puts
            else:  # Call side
                skew_impact = -vol_skew * 0.5  # Negative skew hurts calls
            skew_impacts.append(skew_impact)

        total_skew_impact = sum(skew_impacts)

        return {
            "vol_skew": vol_skew,
            "total_skew_impact": total_skew_impact,
            "put_advantage": vol_skew > 0.02,  # 2% skew threshold
            "call_advantage": vol_skew < -0.02,
            "skew_magnitude": abs(vol_skew)
        }


class LongStraddleStrategy(BaseStrategy):
    """
    Long Straddle options strategy implementation.

    Long Straddle consists of:
    - Long call at strike price (purchased)
    - Long put at same strike price (purchased)

    Structure: Buy Call @ X, Buy Put @ X (same strike, same expiration)

    Volatility strategy with unlimited upside and limited downside.
    Net debit strategy - pay premium upfront for both legs.

    Max profit: Unlimited (as underlying moves far from strike)
    Max loss: Total premium paid (at strike price at expiration)
    Breakeven: Strike ± total premium paid (two breakeven points)

    Best market conditions:
    - Low Volatility regime (4) - enter before volatility expansion
    - High implied volatility rank expected to increase further
    - Expect large price movement in either direction
    - Events or catalysts that may cause significant moves
    """

    def _create_metadata(self) -> StrategyMetadata:
        """Create Long Straddle strategy metadata."""
        return StrategyMetadata(
            name="Long Straddle",
            category=StrategyCategory.VOLATILITY,
            risk_level=RiskLevel.MEDIUM,
            capital_requirement=1.4,  # Higher due to buying both call and put
            description="Long volatility play buying call and put at same strike for large moves either direction",
            typical_market_conditions=[
                "Low volatility regime (4) before expansion",
                "Anticipating high volatility events",
                "Uncertain direction but expecting large moves",
                "Pre-earnings or catalyst plays"
            ]
        )

    def get_strategy_type(self) -> StrategyType:
        """Get strategy type enum."""
        return StrategyType.LONG_STRADDLE

    def validate_market_conditions(self, conditions: MarketConditions) -> bool:
        """
        Validate market conditions for Long Straddle strategy.

        Args:
            conditions: Current market conditions

        Returns:
            True if conditions are suitable for Long Straddle
        """
        # Favorable regime: Low volatility (4) before expansion
        if conditions.regime != 4:
            return False

        # Need potential for volatility expansion (currently low but not too low)
        if conditions.volatility_rank > 0.6:  # Already high volatility
            return False

        if conditions.volatility_rank < 0.1:  # Too low, unlikely to expand quickly
            return False

        # Check time to expiration (need enough time for movement, 15-60 DTE)
        if conditions.time_to_expiration < 10 or conditions.time_to_expiration > 90:
            return False

        return True

    def calculate_entry_criteria(self, conditions: MarketConditions) -> EntrySignal:
        """
        Calculate entry signal for Long Straddle.

        Args:
            conditions: Current market conditions

        Returns:
            EntrySignal with entry decision and confidence
        """
        if not self.validate_market_conditions(conditions):
            return EntrySignal(
                should_enter=False,
                confidence=0.0,
                reasons=["Market conditions not suitable for Long Straddle"]
            )

        confidence = 0.0
        reasons = []

        # Regime scoring (entering in low vol before expansion)
        if conditions.regime == 4:  # Perfect regime
            confidence += 0.3
            reasons.append("Low volatility regime 4 optimal for long straddle entry")

        # Volatility scoring (want low but not too low)
        if 0.2 <= conditions.volatility_rank <= 0.4:  # Sweet spot
            confidence += 0.25
            reasons.append(f"Volatility rank {conditions.volatility_rank:.2f} ideal for volatility expansion play")
        elif 0.1 <= conditions.volatility_rank < 0.2:  # Lower but acceptable
            confidence += 0.15
            reasons.append("Low volatility provides good entry point")
        elif 0.4 < conditions.volatility_rank <= 0.6:  # Higher but still workable
            confidence += 0.1
            reasons.append("Moderate volatility still allows for expansion")

        # Time to expiration scoring
        if 20 <= conditions.time_to_expiration <= 45:  # Optimal range
            confidence += 0.2
            reasons.append(f"Optimal {conditions.time_to_expiration} DTE for volatility play")
        elif 10 <= conditions.time_to_expiration < 20:  # Shorter term
            confidence += 0.1
            reasons.append("Shorter timeframe acceptable for near-term catalysts")
        elif 45 < conditions.time_to_expiration <= 90:  # Longer term
            confidence += 0.15
            reasons.append("Longer timeframe allows for volatility development")

        # Trend uncertainty (good for straddles as direction doesn't matter)
        if abs(conditions.trend_strength) < 0.3:  # Uncertain direction
            confidence += 0.15
            reasons.append("Uncertain trend direction favors straddle strategy")
        elif abs(conditions.trend_strength) < 0.5:  # Some direction but not strong
            confidence += 0.1
            reasons.append("Moderate trend still allows for volatility plays")

        # Vega considerations (want positive vega exposure in low vol environment)
        if conditions.volatility_rank < 0.3:
            confidence += 0.1
            reasons.append("Low volatility environment favors long vega positions")

        # Position sizing recommendation
        recommended_size = 1.0
        if confidence > 0.8:
            recommended_size = 1.2  # Increase size for high confidence
        elif confidence < 0.5:
            recommended_size = 0.7  # Conservative size for lower confidence

        return EntrySignal(
            should_enter=confidence >= 0.6,
            confidence=confidence,
            reasons=reasons,
            recommended_size=recommended_size
        )

    def calculate_exit_criteria(self, position: 'Position', conditions: MarketConditions) -> ExitSignal:
        """
        Calculate exit signal for existing Long Straddle position.

        Args:
            position: Current Long Straddle position
            conditions: Current market conditions

        Returns:
            ExitSignal with exit decision and reasoning
        """
        reasons = []
        urgency = 0.0
        exit_type = ""

        # Time decay considerations
        days_to_exp = position.days_to_expiration
        if days_to_exp <= 7:
            urgency += 0.8
            exit_type = "time"
            reasons.append("Time decay accelerating - high theta risk")
        elif days_to_exp <= 14:
            urgency += 0.4
            exit_type = "time"
            reasons.append("Approaching expiration - theta becoming significant")

        # Profit target exit (volatility expansion realized)
        current_pnl = position.calculate_unrealized_pnl()
        max_loss = position.calculate_max_loss()  # Premium paid

        if max_loss > 0:
            profit_percent = current_pnl / max_loss
            if profit_percent >= 1.0:  # 100% profit (doubled investment)
                urgency += 0.7
                exit_type = "profit"
                reasons.append(f"Excellent profit: {profit_percent:.1%} of premium")
            elif profit_percent >= 0.5:  # 50% profit
                urgency += 0.4
                exit_type = "profit"
                reasons.append(f"Good profit level: {profit_percent:.1%} of premium")
            elif profit_percent >= 0.25:  # 25% profit
                urgency += 0.2
                if not exit_type:
                    exit_type = "profit"
                reasons.append(f"Partial profit available: {profit_percent:.1%} of premium")

        # Loss limit exit (theta decay eating premium)
        if max_loss > 0:
            loss_percent = abs(current_pnl) / max_loss
            if current_pnl < 0 and loss_percent >= 0.7:  # 70% of premium lost
                urgency = 0.9
                exit_type = "loss"
                reasons.append("Majority of premium lost to time decay")
            elif current_pnl < 0 and loss_percent >= 0.5:  # 50% of premium lost
                urgency += 0.6
                exit_type = "loss"
                reasons.append("Significant premium erosion from theta")

        # Volatility considerations
        if conditions.volatility_rank > 0.8:  # Very high volatility achieved
            urgency += 0.3
            if not exit_type:
                exit_type = "profit"
            reasons.append("High volatility achieved - consider profit taking")
        elif conditions.volatility_rank < 0.15:  # Volatility crushed
            urgency += 0.5
            if not exit_type:
                exit_type = "loss"
            reasons.append("Volatility crush working against position")

        # Regime change away from favorable conditions
        unfavorable_regimes = [3, 8]  # High vol and crisis (already realized)
        if conditions.regime in unfavorable_regimes:
            if current_pnl > 0:
                urgency += 0.4
                exit_type = "profit"
                reasons.append(f"High volatility regime {conditions.regime} - take profits")
            else:
                urgency += 0.3
                reasons.append(f"Regime {conditions.regime} may limit further expansion")

        # Movement analysis - check if significant move occurred
        strikes = getattr(position, 'strikes', None)
        current_price = getattr(position, 'current_underlying_price', None)
        if isinstance(strikes, (list, tuple)) and strikes and current_price is not None:
            strike = strikes[0]  # Straddle uses same strike for both legs
            distance_from_strike = abs(current_price - strike) / strike
            if distance_from_strike > 0.1:  # Moved more than 10% from strike
                urgency += 0.3
                if current_pnl > 0:
                    exit_type = "profit"
                    reasons.append("Significant price movement achieved profit")
                else:
                    reasons.append("Large move occurred but may need more time")

        # Vega risk management
        if conditions.volatility_rank > 0.9 and days_to_exp > 14:
            urgency += 0.2
            reasons.append("Extreme volatility - vega risk increasing")

        return ExitSignal(
            should_exit=urgency >= 0.6,
            urgency=min(urgency, 1.0),
            exit_type=exit_type or "hold",
            reasons=reasons or ["Position within acceptable parameters"]
        )

    def get_risk_metrics(self, strikes: List[int], underlying_price: int,
                        time_to_expiration: int, volatility: float) -> RiskMetrics:
        """
        Calculate risk metrics for Long Straddle.

        Args:
            strikes: [strike_price] in cents (same for both call and put)
            underlying_price: Current underlying price in cents
            time_to_expiration: Days to expiration
            volatility: Implied volatility

        Returns:
            RiskMetrics for the Long Straddle
        """
        if len(strikes) != 1:
            raise ValueError("Long Straddle requires exactly 1 strike price")

        strike = strikes[0]

        # Estimate total premium cost (simplified - would use Black-Scholes in production)
        # ATM straddle typically costs 3-8% of underlying price depending on volatility and time
        time_factor = min(time_to_expiration / 30, 2.0)  # More time = higher premium
        vol_factor = volatility * 2  # Higher vol = higher premium
        atm_factor = 1.0 + abs(underlying_price - strike) / underlying_price * 2  # Distance from ATM

        premium_percentage = (0.04 + vol_factor * 0.03) * time_factor * atm_factor
        total_premium = int(underlying_price * premium_percentage)

        # Risk calculations
        max_profit = float('inf')  # Unlimited profit potential
        max_loss = total_premium  # Limited to premium paid

        # Breakeven points
        breakeven_upper = strike + total_premium
        breakeven_lower = strike - total_premium
        breakevens = [breakeven_lower, breakeven_upper]

        # Probability estimation (simplified)
        # Estimate probability price ends outside breakeven range
        move_needed_percent = total_premium / strike  # % move needed in either direction

        # Rough estimation based on normal distribution and volatility
        # Higher volatility and more time increase probability of profit
        base_prob = min(0.4, volatility * 0.6)  # Base probability from volatility
        time_prob = min(0.2, time_to_expiration / 100)  # Time component
        profit_probability = min(base_prob + time_prob, 0.7)

        # Risk/reward ratio (difficult to calculate with unlimited upside)
        # Use expected profit vs max loss for practical purposes
        expected_profit = total_premium * 2  # Assume 200% return target
        risk_reward_ratio = expected_profit / max_loss if max_loss > 0 else 2.0

        # Capital requirements
        capital_requirement = max_loss  # Full premium paid upfront
        margin_requirement = int(max_loss * 1.1)  # Small buffer for debit strategy

        return RiskMetrics(
            max_profit=2147483647,  # Max int value to represent unlimited
            max_loss=max_loss,
            breakeven_points=breakevens,
            profit_probability=profit_probability,
            risk_reward_ratio=risk_reward_ratio,
            capital_requirement=capital_requirement,
            margin_requirement=margin_requirement
        )

    def get_position_legs(self, strikes: List[int], expiration_date: datetime) -> List[PositionLeg]:
        """
        Get position legs for Long Straddle construction.

        Args:
            strikes: [strike_price] in cents (same for both call and put)
            expiration_date: Option expiration date

        Returns:
            List of PositionLeg objects for Long Straddle
        """
        if len(strikes) != 1:
            raise ValueError("Long Straddle requires exactly 1 strike: [strike_price]")

        strike = strikes[0]

        return [
            # Long call at strike
            PositionLeg(
                option_type=OptionType.CALL,
                strike=strike,
                quantity=1,  # Long position
                expiration_date=expiration_date
            ),
            # Long put at same strike
            PositionLeg(
                option_type=OptionType.PUT,
                strike=strike,
                quantity=1,  # Long position
                expiration_date=expiration_date
            )
        ]

    def _validate_strategy_strikes(self, strikes: List[int], underlying_price: int) -> bool:
        """
        Validate Long Straddle strike configuration.

        Args:
            strikes: Strike prices to validate
            underlying_price: Current underlying price

        Returns:
            True if strikes are valid for Long Straddle
        """
        if len(strikes) != 1:
            return False

        strike = strikes[0]

        # Strike should be close to current price (ATM or near ATM)
        # Allow up to 10% away from ATM for flexibility
        distance_from_atm = abs(strike - underlying_price) / underlying_price
        if distance_from_atm > 0.10:
            return False

        return True


class ShortStraddleStrategy(BaseStrategy):
    """
    Short Straddle options strategy implementation.

    Short Straddle consists of:
    - Short call at strike price (sold)
    - Short put at same strike price (sold)

    Structure: Sell Call @ X, Sell Put @ X (same strike, same expiration)

    Volatility strategy with limited upside and unlimited downside risk.
    Net credit strategy - receive premium upfront for both legs.

    Max profit: Total premium received (at strike price at expiration)
    Max loss: Unlimited (as underlying moves far from strike)
    Breakeven: Strike ± total premium received (two breakeven points)

    Best market conditions:
    - High Volatility regime (3) - enter expecting volatility contraction
    - High implied volatility rank expected to decrease
    - Expect sideways/range-bound movement near strike
    - After major events when volatility typically declines
    """

    def _create_metadata(self) -> StrategyMetadata:
        """Create Short Straddle strategy metadata."""
        return StrategyMetadata(
            name="Short Straddle",
            category=StrategyCategory.VOLATILITY,
            risk_level=RiskLevel.VERY_HIGH,
            capital_requirement=2.5,  # Very high due to unlimited risk and margin requirements
            description="Short volatility play selling call and put at same strike for range-bound movement",
            typical_market_conditions=[
                "High volatility regime (3) expecting contraction",
                "Post-event volatility decline expected",
                "Range-bound price action anticipated",
                "High implied volatility crush scenarios"
            ]
        )

    def get_strategy_type(self) -> StrategyType:
        """Get strategy type enum."""
        return StrategyType.SHORT_STRADDLE

    def validate_market_conditions(self, conditions: MarketConditions) -> bool:
        """
        Validate market conditions for Short Straddle strategy.

        Args:
            conditions: Current market conditions

        Returns:
            True if conditions are suitable for Short Straddle
        """
        # Favorable regime: High volatility (3) expecting contraction
        if conditions.regime != 3:
            return False

        # Need high volatility that can contract
        if conditions.volatility_rank < 0.6:  # Need high volatility to make it worthwhile
            return False

        # Check time to expiration (shorter is better for theta decay, 10-45 DTE)
        if conditions.time_to_expiration < 7 or conditions.time_to_expiration > 60:
            return False

        # Avoid in crisis regime (8) due to extreme unpredictability
        if conditions.regime == 8:
            return False

        return True

    def calculate_entry_criteria(self, conditions: MarketConditions) -> EntrySignal:
        """
        Calculate entry signal for Short Straddle.

        Args:
            conditions: Current market conditions

        Returns:
            EntrySignal with entry decision and confidence
        """
        if not self.validate_market_conditions(conditions):
            return EntrySignal(
                should_enter=False,
                confidence=0.0,
                reasons=["Market conditions not suitable for Short Straddle"]
            )

        confidence = 0.0
        reasons = []

        # Regime scoring (entering in high vol expecting contraction)
        if conditions.regime == 3:  # Perfect regime
            confidence += 0.3
            reasons.append("High volatility regime 3 optimal for short straddle entry")

        # Volatility scoring (want high volatility to collect premium)
        if conditions.volatility_rank >= 0.8:  # Very high volatility
            confidence += 0.3
            reasons.append(f"Very high volatility rank {conditions.volatility_rank:.2f} excellent for premium collection")
        elif conditions.volatility_rank >= 0.6:  # High volatility
            confidence += 0.2
            reasons.append("High volatility provides good premium collection opportunity")

        # Time to expiration scoring (shorter is better for theta)
        if 10 <= conditions.time_to_expiration <= 30:  # Optimal range
            confidence += 0.2
            reasons.append(f"Optimal {conditions.time_to_expiration} DTE for theta decay benefit")
        elif 7 <= conditions.time_to_expiration < 10:  # Very short term
            confidence += 0.15
            reasons.append("Short timeframe maximizes theta decay")
        elif 30 < conditions.time_to_expiration <= 45:  # Longer but acceptable
            confidence += 0.1
            reasons.append("Longer timeframe still provides theta benefit")

        # Trend considerations (want low trend strength for range-bound movement)
        if abs(conditions.trend_strength) < 0.2:  # Very sideways
            confidence += 0.15
            reasons.append("Weak trend favors range-bound movement")
        elif abs(conditions.trend_strength) < 0.4:  # Some trend but manageable
            confidence += 0.05
            reasons.append("Moderate trend acceptable for wide strikes")

        # Risk management considerations (be more conservative)
        if conditions.volatility_rank > 0.95:  # Extreme volatility
            confidence -= 0.1
            reasons.append("Extreme volatility increases assignment and gap risk")

        # Position sizing recommendation (very conservative due to unlimited risk)
        recommended_size = 0.5  # Start very conservative
        if confidence > 0.8:
            recommended_size = 0.8  # Still conservative even with high confidence
        elif confidence < 0.6:
            recommended_size = 0.3  # Very small position for lower confidence

        return EntrySignal(
            should_enter=confidence >= 0.7,  # Higher threshold due to unlimited risk
            confidence=confidence,
            reasons=reasons,
            recommended_size=recommended_size
        )

    def calculate_exit_criteria(self, position: 'Position', conditions: MarketConditions) -> ExitSignal:
        """
        Calculate exit signal for existing Short Straddle position.

        Args:
            position: Current Short Straddle position
            conditions: Current market conditions

        Returns:
            ExitSignal with exit decision and reasoning
        """
        reasons = []
        urgency = 0.0
        exit_type = ""

        # Assignment risk management (very important for short strategies)
        days_to_exp = position.days_to_expiration
        if days_to_exp <= 3:
            urgency += 0.9
            exit_type = "time"
            reasons.append("Extreme assignment risk - close immediately")
        elif days_to_exp <= 7:
            urgency += 0.6
            exit_type = "time"
            reasons.append("High assignment risk approaching expiration")

        # Profit target exit (theta decay working)
        current_pnl = position.calculate_unrealized_pnl()
        max_profit = position.calculate_max_profit()  # Premium received

        if max_profit > 0:
            profit_percent = current_pnl / max_profit
            if profit_percent >= 0.5:  # 50% of max profit (aggressive for unlimited risk)
                urgency += 0.7
                exit_type = "profit"
                reasons.append(f"Good profit target reached: {profit_percent:.1%} of premium")
            elif profit_percent >= 0.25:  # 25% profit
                urgency += 0.4
                exit_type = "profit"
                reasons.append(f"Partial profit from theta decay: {profit_percent:.1%}")

        # Loss limit exit (unlimited risk management)
        if max_profit > 0:
            loss_ratio = abs(current_pnl) / max_profit
            if current_pnl < 0 and loss_ratio >= 2.0:  # Lost 200% of premium
                urgency = 1.0
                exit_type = "loss"
                reasons.append("Severe loss - close immediately")
            elif current_pnl < 0 and loss_ratio >= 1.0:  # Lost all premium collected
                urgency += 0.8
                exit_type = "loss"
                reasons.append("All premium lost - significant loss developing")
            elif current_pnl < 0 and loss_ratio >= 0.5:  # Lost 50% of premium
                urgency += 0.5
                exit_type = "loss"
                reasons.append("Significant loss developing")

        # Volatility expansion (very bad for short straddle)
        if conditions.volatility_rank > 0.9:
            urgency += 0.6
            if not exit_type:
                exit_type = "adjustment"
            reasons.append("Volatility expansion threatening position")
        elif conditions.volatility_rank < 0.3:  # Mission accomplished
            urgency += 0.3
            if current_pnl > 0:
                exit_type = "profit"
                reasons.append("Volatility contraction achieved - take profits")

        # Regime change to unfavorable conditions
        unfavorable_regimes = [8, 1, 2]  # Crisis, strong bull/bear trends
        if conditions.regime in unfavorable_regimes:
            urgency += 0.5
            if not exit_type:
                exit_type = "adjustment"
            reasons.append(f"Regime {conditions.regime} dangerous for short straddle")

        # Price movement analysis (critical for unlimited risk strategy)
        if len(position.strikes) >= 1:
            strike = position.strikes[0]  # Straddle uses same strike
            current_price = position.current_underlying_price

            distance_from_strike = abs(current_price - strike) / strike
            if distance_from_strike > 0.15:  # Moved more than 15% from strike
                urgency += 0.8
                exit_type = "loss"
                reasons.append("Price moved significantly from strike - high loss risk")
            elif distance_from_strike > 0.1:  # Moved more than 10%
                urgency += 0.5
                if not exit_type:
                    exit_type = "adjustment"
                reasons.append("Price movement approaching danger zone")

        # Trend strength (strong trends dangerous for short straddle)
        if abs(conditions.trend_strength) > 0.6:
            urgency += 0.4
            if not exit_type:
                exit_type = "adjustment"
            reasons.append("Strong trend developing - dangerous for short straddle")

        return ExitSignal(
            should_exit=urgency >= 0.5,  # More aggressive exit due to unlimited risk
            urgency=min(urgency, 1.0),
            exit_type=exit_type or "hold",
            reasons=reasons or ["Position within acceptable parameters"]
        )

    def get_risk_metrics(self, strikes: List[int], underlying_price: int,
                        time_to_expiration: int, volatility: float) -> RiskMetrics:
        """
        Calculate risk metrics for Short Straddle.

        Args:
            strikes: [strike_price] in cents (same for both call and put)
            underlying_price: Current underlying price in cents
            time_to_expiration: Days to expiration
            volatility: Implied volatility

        Returns:
            RiskMetrics for the Short Straddle
        """
        if len(strikes) != 1:
            raise ValueError("Short Straddle requires exactly 1 strike price")

        strike = strikes[0]

        # Estimate total premium collected (simplified)
        # Use similar calculation as Long Straddle but received instead of paid
        time_factor = min(time_to_expiration / 30, 2.0)
        vol_factor = volatility * 2
        atm_factor = 1.0 + abs(underlying_price - strike) / underlying_price * 2

        premium_percentage = (0.04 + vol_factor * 0.03) * time_factor * atm_factor
        total_premium = int(underlying_price * premium_percentage)

        # Risk calculations
        max_profit = total_premium  # Limited to premium collected
        max_loss = 2147483647  # Max int value to represent unlimited loss

        # Breakeven points
        breakeven_upper = strike + total_premium
        breakeven_lower = strike - total_premium
        breakevens = [breakeven_lower, breakeven_upper]

        # Probability estimation (need price to stay between breakevens)
        move_needed_percent = total_premium / strike

        # Probability decreases with volatility and time
        # Short straddle profits from low movement
        base_prob = max(0.2, 0.6 - volatility * 0.4)  # Lower prob with higher vol
        time_prob = max(0.1, 0.3 - time_to_expiration / 100)  # Lower prob with more time
        profit_probability = max(base_prob + time_prob, 0.3)

        # Risk/reward ratio (limited profit, unlimited loss)
        # Use a conservative expected loss estimate
        expected_loss = total_premium * 3  # Conservative loss estimate
        risk_reward_ratio = max_profit / expected_loss if expected_loss > 0 else 0.33

        # Capital requirements (much higher due to margin requirements)
        # Short straddle requires significant margin for unlimited risk
        margin_per_contract = underlying_price * 20 // 100  # 20% of underlying
        capital_requirement = margin_per_contract * 2  # For both call and put
        margin_requirement = int(capital_requirement * 1.5)  # 50% buffer for volatility

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
        Get position legs for Short Straddle construction.

        Args:
            strikes: [strike_price] in cents (same for both call and put)
            expiration_date: Option expiration date

        Returns:
            List of PositionLeg objects for Short Straddle
        """
        if len(strikes) != 1:
            raise ValueError("Short Straddle requires exactly 1 strike: [strike_price]")

        strike = strikes[0]

        return [
            # Short call at strike
            PositionLeg(
                option_type=OptionType.CALL,
                strike=strike,
                quantity=-1,  # Short position
                expiration_date=expiration_date
            ),
            # Short put at same strike
            PositionLeg(
                option_type=OptionType.PUT,
                strike=strike,
                quantity=-1,  # Short position
                expiration_date=expiration_date
            )
        ]

    def _validate_strategy_strikes(self, strikes: List[int], underlying_price: int) -> bool:
        """
        Validate Short Straddle strike configuration.

        Args:
            strikes: Strike prices to validate
            underlying_price: Current underlying price

        Returns:
            True if strikes are valid for Short Straddle
        """
        if len(strikes) != 1:
            return False

        strike = strikes[0]

        # Strike should be close to current price (ATM or near ATM)
        # Allow up to 5% away from ATM for short straddle (tighter than long)
        distance_from_atm = abs(strike - underlying_price) / underlying_price
        if distance_from_atm > 0.05:
            return False

        return True


class LongStrangleStrategy(BaseStrategy):
    """
    Long Strangle options strategy implementation.

    Long Strangle consists of:
    - Long call at higher strike (OTM call)
    - Long put at lower strike (OTM put)

    Structure: Buy Put @ A, Buy Call @ B (A < current price < B, same expiration)

    Volatility strategy with unlimited upside and limited downside.
    Net debit strategy - pay premium upfront for both legs.
    Cheaper than straddle but wider breakeven range.

    Max profit: Unlimited (as underlying moves far from either strike)
    Max loss: Total premium paid (between strikes at expiration)
    Breakeven: Put strike - total premium, Call strike + total premium

    Best market conditions:
    - Low Volatility regime (4) - enter before volatility expansion
    - Lower cost alternative to straddle
    - Expect large price movement in either direction
    - When straddle is too expensive due to high ATM premiums
    """

    def _create_metadata(self) -> StrategyMetadata:
        """Create Long Strangle strategy metadata."""
        return StrategyMetadata(
            name="Long Strangle",
            category=StrategyCategory.VOLATILITY,
            risk_level=RiskLevel.MEDIUM,
            capital_requirement=1.2,  # Lower than straddle due to OTM options
            description="Long volatility play with OTM call and put for cheaper premium and wider breakeven range",
            typical_market_conditions=[
                "Low volatility regime (4) before expansion",
                "Cost-conscious volatility plays",
                "Expecting very large moves in either direction",
                "When ATM straddle premiums are too high"
            ]
        )

    def get_strategy_type(self) -> StrategyType:
        """Get strategy type enum."""
        return StrategyType.LONG_STRANGLE

    def validate_market_conditions(self, conditions: MarketConditions) -> bool:
        """
        Validate market conditions for Long Strangle strategy.

        Args:
            conditions: Current market conditions

        Returns:
            True if conditions are suitable for Long Strangle
        """
        # Same regime requirements as Long Straddle
        if conditions.regime != 4:
            return False

        # Need potential for volatility expansion
        if conditions.volatility_rank > 0.6:  # Already high volatility
            return False

        if conditions.volatility_rank < 0.1:  # Too low, unlikely to expand quickly
            return False

        # Check time to expiration (need more time than straddle for larger moves)
        if conditions.time_to_expiration < 15 or conditions.time_to_expiration > 120:
            return False

        return True

    def calculate_entry_criteria(self, conditions: MarketConditions) -> EntrySignal:
        """
        Calculate entry signal for Long Strangle.

        Args:
            conditions: Current market conditions

        Returns:
            EntrySignal with entry decision and confidence
        """
        if not self.validate_market_conditions(conditions):
            return EntrySignal(
                should_enter=False,
                confidence=0.0,
                reasons=["Market conditions not suitable for Long Strangle"]
            )

        confidence = 0.0
        reasons = []

        # Regime scoring
        if conditions.regime == 4:  # Perfect regime
            confidence += 0.25
            reasons.append("Low volatility regime 4 optimal for long strangle entry")

        # Volatility scoring (similar to straddle but slightly more forgiving)
        if 0.2 <= conditions.volatility_rank <= 0.4:  # Sweet spot
            confidence += 0.3
            reasons.append(f"Volatility rank {conditions.volatility_rank:.2f} ideal for volatility expansion play")
        elif 0.1 <= conditions.volatility_rank < 0.2:  # Lower but acceptable
            confidence += 0.2
            reasons.append("Low volatility provides good entry point for strangle")
        elif 0.4 < conditions.volatility_rank <= 0.6:  # Higher but still workable
            confidence += 0.15
            reasons.append("Moderate volatility still allows for expansion")

        # Time to expiration scoring (prefer longer timeframes than straddle)
        if 30 <= conditions.time_to_expiration <= 60:  # Optimal range
            confidence += 0.2
            reasons.append(f"Optimal {conditions.time_to_expiration} DTE for strangle volatility play")
        elif 60 < conditions.time_to_expiration <= 90:  # Longer term - good for strangles
            confidence += 0.15
            reasons.append("Longer timeframe good for large moves needed")
        elif 15 <= conditions.time_to_expiration < 30:  # Shorter term
            confidence += 0.1
            reasons.append("Shorter timeframe acceptable for near-term catalysts")

        # Trend uncertainty (even better for strangles as they need larger moves)
        if abs(conditions.trend_strength) < 0.2:  # Uncertain direction
            confidence += 0.15
            reasons.append("Uncertain trend direction ideal for strangle strategy")
        elif abs(conditions.trend_strength) < 0.5:  # Some direction
            confidence += 0.1
            reasons.append("Moderate trend acceptable for wide strangle strikes")

        # Cost advantage consideration
        if conditions.volatility_rank < 0.3:
            confidence += 0.1
            reasons.append("Lower volatility makes strangle cost-effective vs straddle")

        # Position sizing recommendation
        recommended_size = 1.0
        if confidence > 0.8:
            recommended_size = 1.3  # Can be more aggressive due to lower cost
        elif confidence < 0.5:
            recommended_size = 0.8  # Conservative size for lower confidence

        return EntrySignal(
            should_enter=confidence >= 0.6,
            confidence=confidence,
            reasons=reasons,
            recommended_size=recommended_size
        )

    def calculate_exit_criteria(self, position: 'Position', conditions: MarketConditions) -> ExitSignal:
        """
        Calculate exit signal for existing Long Strangle position.

        Args:
            position: Current Long Strangle position
            conditions: Current market conditions

        Returns:
            ExitSignal with exit decision and reasoning
        """
        reasons = []
        urgency = 0.0
        exit_type = ""

        # Time decay considerations (similar to straddle but less aggressive)
        days_to_exp = position.days_to_expiration
        if days_to_exp <= 10:
            urgency += 0.7
            exit_type = "time"
            reasons.append("Time decay accelerating - high theta risk")
        elif days_to_exp <= 21:
            urgency += 0.3
            exit_type = "time"
            reasons.append("Approaching expiration - theta becoming significant")

        # Profit target exit
        current_pnl = position.calculate_unrealized_pnl()
        max_loss = position.calculate_max_loss()  # Premium paid

        if max_loss > 0:
            profit_percent = current_pnl / max_loss
            if profit_percent >= 1.5:  # 150% profit (need larger moves)
                urgency += 0.7
                exit_type = "profit"
                reasons.append(f"Excellent profit: {profit_percent:.1%} of premium")
            elif profit_percent >= 0.75:  # 75% profit
                urgency += 0.4
                exit_type = "profit"
                reasons.append(f"Good profit level: {profit_percent:.1%} of premium")
            elif profit_percent >= 0.3:  # 30% profit
                urgency += 0.2
                if not exit_type:
                    exit_type = "profit"
                reasons.append(f"Partial profit available: {profit_percent:.1%} of premium")

        # Loss limit exit
        if max_loss > 0:
            loss_percent = abs(current_pnl) / max_loss
            if current_pnl < 0 and loss_percent >= 0.8:  # 80% of premium lost
                urgency = 0.9
                exit_type = "loss"
                reasons.append("Majority of premium lost to time decay")
            elif current_pnl < 0 and loss_percent >= 0.6:  # 60% of premium lost
                urgency += 0.6
                exit_type = "loss"
                reasons.append("Significant premium erosion from theta")

        # Volatility considerations
        if conditions.volatility_rank > 0.85:  # Very high volatility achieved
            urgency += 0.3
            if not exit_type:
                exit_type = "profit"
            reasons.append("High volatility achieved - consider profit taking")
        elif conditions.volatility_rank < 0.15:  # Volatility crushed
            urgency += 0.4
            if not exit_type:
                exit_type = "loss"
            reasons.append("Volatility crush working against position")

        # Regime change considerations
        unfavorable_regimes = [3, 8]  # High vol and crisis
        if conditions.regime in unfavorable_regimes:
            if current_pnl > 0:
                urgency += 0.3
                exit_type = "profit"
                reasons.append(f"High volatility regime {conditions.regime} - take profits")
            else:
                urgency += 0.2
                reasons.append(f"Regime {conditions.regime} may limit further expansion")

        # Movement analysis - need larger moves for strangle profit
        if len(position.strikes) >= 2:
            put_strike = min(position.strikes)
            call_strike = max(position.strikes)
            current_price = position.current_underlying_price

            # Check if moved significantly beyond either strike
            beyond_call = max(0, current_price - call_strike) / call_strike
            beyond_put = max(0, put_strike - current_price) / put_strike

            if beyond_call > 0.05 or beyond_put > 0.05:  # Moved 5% beyond strike
                urgency += 0.3
                if current_pnl > 0:
                    exit_type = "profit"
                    reasons.append("Significant movement beyond strikes achieved profit")
                else:
                    reasons.append("Large move occurred but may need more development")

        return ExitSignal(
            should_exit=urgency >= 0.6,
            urgency=min(urgency, 1.0),
            exit_type=exit_type or "hold",
            reasons=reasons or ["Position within acceptable parameters"]
        )

    def get_risk_metrics(self, strikes: List[int], underlying_price: int,
                        time_to_expiration: int, volatility: float) -> RiskMetrics:
        """
        Calculate risk metrics for Long Strangle.

        Args:
            strikes: [put_strike, call_strike] in cents (put_strike < call_strike)
            underlying_price: Current underlying price in cents
            time_to_expiration: Days to expiration
            volatility: Implied volatility

        Returns:
            RiskMetrics for the Long Strangle
        """
        if len(strikes) != 2:
            raise ValueError("Long Strangle requires exactly 2 strikes: [put_strike, call_strike]")

        put_strike, call_strike = strikes

        if put_strike >= call_strike:
            raise ValueError("Put strike must be lower than call strike")

        # Estimate premium for OTM options (typically cheaper than ATM)
        time_factor = min(time_to_expiration / 30, 2.0)
        vol_factor = volatility * 1.8  # Slightly less than straddle

        # Distance from ATM affects premium (OTM options are cheaper)
        put_otm_factor = max(0.3, 1.0 - (underlying_price - put_strike) / underlying_price)
        call_otm_factor = max(0.3, 1.0 - (call_strike - underlying_price) / underlying_price)

        put_premium = int(underlying_price * 0.025 * vol_factor * time_factor * put_otm_factor)
        call_premium = int(underlying_price * 0.025 * vol_factor * time_factor * call_otm_factor)
        total_premium = put_premium + call_premium

        # Risk calculations
        max_profit = float('inf')  # Unlimited profit potential
        max_loss = total_premium  # Limited to premium paid

        # Breakeven points (wider than straddle)
        breakeven_lower = put_strike - total_premium
        breakeven_upper = call_strike + total_premium
        breakevens = [breakeven_lower, breakeven_upper]

        # Probability estimation (lower than straddle due to wider range needed)
        # Need price to move beyond either breakeven
        lower_move_needed = (underlying_price - breakeven_lower) / underlying_price
        upper_move_needed = (breakeven_upper - underlying_price) / underlying_price
        min_move_needed = min(lower_move_needed, upper_move_needed)

        # Probability decreases with wider ranges needed
        base_prob = min(0.35, volatility * 0.5)  # Lower than straddle
        time_prob = min(0.15, time_to_expiration / 120)  # More time helps strangles
        profit_probability = min(base_prob + time_prob, 0.6)

        # Risk/reward ratio
        expected_profit = total_premium * 2.5  # Higher target due to wider breakevens
        risk_reward_ratio = expected_profit / max_loss if max_loss > 0 else 2.5

        # Capital requirements
        capital_requirement = max_loss  # Full premium paid upfront
        margin_requirement = int(max_loss * 1.1)  # Small buffer for debit strategy

        return RiskMetrics(
            max_profit=2147483647,  # Max int value to represent unlimited
            max_loss=max_loss,
            breakeven_points=breakevens,
            profit_probability=profit_probability,
            risk_reward_ratio=risk_reward_ratio,
            capital_requirement=capital_requirement,
            margin_requirement=margin_requirement
        )

    def get_position_legs(self, strikes: List[int], expiration_date: datetime) -> List[PositionLeg]:
        """
        Get position legs for Long Strangle construction.

        Args:
            strikes: [put_strike, call_strike] in cents (put_strike < call_strike)
            expiration_date: Option expiration date

        Returns:
            List of PositionLeg objects for Long Strangle
        """
        if len(strikes) != 2:
            raise ValueError("Long Strangle requires exactly 2 strikes: [put_strike, call_strike]")

        put_strike, call_strike = strikes

        return [
            # Long put at lower strike (OTM)
            PositionLeg(
                option_type=OptionType.PUT,
                strike=put_strike,
                quantity=1,  # Long position
                expiration_date=expiration_date
            ),
            # Long call at higher strike (OTM)
            PositionLeg(
                option_type=OptionType.CALL,
                strike=call_strike,
                quantity=1,  # Long position
                expiration_date=expiration_date
            )
        ]

    def _validate_strategy_strikes(self, strikes: List[int], underlying_price: int) -> bool:
        """
        Validate Long Strangle strike configuration.

        Args:
            strikes: Strike prices to validate
            underlying_price: Current underlying price

        Returns:
            True if strikes are valid for Long Strangle
        """
        if len(strikes) != 2:
            return False

        put_strike, call_strike = strikes

        # Put strike must be lower than call strike
        if put_strike >= call_strike:
            return False

        # Current price should be between the strikes
        if not (put_strike < underlying_price < call_strike):
            return False

        # Strikes should be reasonably OTM but not too far (5% to 20% OTM)
        put_distance = (underlying_price - put_strike) / underlying_price
        call_distance = (call_strike - underlying_price) / underlying_price

        if put_distance < 0.05 or put_distance > 0.20:  # Put too close or too far OTM
            return False

        if call_distance < 0.05 or call_distance > 0.20:  # Call too close or too far OTM
            return False

        # Strikes should be reasonably balanced
        if abs(put_distance - call_distance) > 0.10:  # More than 10% difference
            return False

        return True


class ShortStrangleStrategy(BaseStrategy):
    """
    Short Strangle options strategy implementation.

    Short Strangle consists of:
    - Short call at higher strike (OTM call sold)
    - Short put at lower strike (OTM put sold)

    Structure: Sell Put @ A, Sell Call @ B (A < current price < B, same expiration)

    Volatility strategy with limited upside and unlimited downside risk.
    Net credit strategy - receive premium upfront for both legs.
    Wider profit zone than short straddle.

    Max profit: Total premium received (between strikes at expiration)
    Max loss: Unlimited (as underlying moves far beyond either strike)
    Breakeven: Put strike - total premium, Call strike + total premium

    Best market conditions:
    - High Volatility regime (3) - enter expecting volatility contraction
    - Expecting range-bound movement between strikes
    - Wider profit zone than short straddle
    - When short straddle strikes are too tight
    """

    def _create_metadata(self) -> StrategyMetadata:
        """Create Short Strangle strategy metadata."""
        return StrategyMetadata(
            name="Short Strangle",
            category=StrategyCategory.VOLATILITY,
            risk_level=RiskLevel.VERY_HIGH,
            capital_requirement=2.0,  # High but less than short straddle due to wider zone
            description="Short volatility play with wider profit zone selling OTM call and put",
            typical_market_conditions=[
                "High volatility regime (3) expecting contraction",
                "Range-bound movement expected between wide strikes",
                "Alternative to short straddle with wider profit zone",
                "Post-event volatility decline scenarios"
            ]
        )

    def get_strategy_type(self) -> StrategyType:
        """Get strategy type enum."""
        return StrategyType.SHORT_STRANGLE

    def validate_market_conditions(self, conditions: MarketConditions) -> bool:
        """
        Validate market conditions for Short Strangle strategy.

        Args:
            conditions: Current market conditions

        Returns:
            True if conditions are suitable for Short Strangle
        """
        # Same regime requirements as Short Straddle
        if conditions.regime != 3:
            return False

        # Need high volatility that can contract
        if conditions.volatility_rank < 0.5:  # Slightly lower threshold than straddle
            return False

        # Check time to expiration
        if conditions.time_to_expiration < 10 or conditions.time_to_expiration > 60:
            return False

        # Avoid crisis regime
        if conditions.regime == 8:
            return False

        return True

    def calculate_entry_criteria(self, conditions: MarketConditions) -> EntrySignal:
        """
        Calculate entry signal for Short Strangle.

        Args:
            conditions: Current market conditions

        Returns:
            EntrySignal with entry decision and confidence
        """
        if not self.validate_market_conditions(conditions):
            return EntrySignal(
                should_enter=False,
                confidence=0.0,
                reasons=["Market conditions not suitable for Short Strangle"]
            )

        confidence = 0.0
        reasons = []

        # Regime scoring
        if conditions.regime == 3:  # Perfect regime
            confidence += 0.25
            reasons.append("High volatility regime 3 optimal for short strangle entry")

        # Volatility scoring (slightly more forgiving than short straddle)
        if conditions.volatility_rank >= 0.8:  # Very high volatility
            confidence += 0.3
            reasons.append(f"Very high volatility rank {conditions.volatility_rank:.2f} excellent for premium collection")
        elif conditions.volatility_rank >= 0.6:  # High volatility
            confidence += 0.25
            reasons.append("High volatility provides good premium collection opportunity")
        elif conditions.volatility_rank >= 0.5:  # Moderate-high volatility
            confidence += 0.15
            reasons.append("Moderate-high volatility acceptable for strangle")

        # Time to expiration scoring
        if 15 <= conditions.time_to_expiration <= 35:  # Optimal range
            confidence += 0.2
            reasons.append(f"Optimal {conditions.time_to_expiration} DTE for theta decay benefit")
        elif 10 <= conditions.time_to_expiration < 15:  # Shorter term
            confidence += 0.15
            reasons.append("Short timeframe maximizes theta decay")
        elif 35 < conditions.time_to_expiration <= 50:  # Longer but acceptable
            confidence += 0.1
            reasons.append("Longer timeframe still provides theta benefit")

        # Trend considerations (wider range tolerance than straddle)
        if abs(conditions.trend_strength) < 0.3:  # Moderate range tolerance
            confidence += 0.15
            reasons.append("Moderate trend acceptable for wide strangle strikes")
        elif abs(conditions.trend_strength) < 0.5:  # Some trend
            confidence += 0.1
            reasons.append("Trend strength manageable with wider strikes")

        # Risk management (more conservative entry)
        if conditions.volatility_rank > 0.95:  # Extreme volatility
            confidence -= 0.05
            reasons.append("Extreme volatility increases gap risk")

        # Position sizing (conservative due to unlimited risk)
        recommended_size = 0.6  # More conservative than short straddle
        if confidence > 0.8:
            recommended_size = 0.9
        elif confidence < 0.6:
            recommended_size = 0.4

        return EntrySignal(
            should_enter=confidence >= 0.65,  # Slightly lower threshold than short straddle
            confidence=confidence,
            reasons=reasons,
            recommended_size=recommended_size
        )

    def calculate_exit_criteria(self, position: 'Position', conditions: MarketConditions) -> ExitSignal:
        """
        Calculate exit signal for existing Short Strangle position.

        Args:
            position: Current Short Strangle position
            conditions: Current market conditions

        Returns:
            ExitSignal with exit decision and reasoning
        """
        reasons = []
        urgency = 0.0
        exit_type = ""

        # Assignment risk management
        days_to_exp = position.days_to_expiration
        if days_to_exp <= 5:
            urgency += 0.8
            exit_type = "time"
            reasons.append("High assignment risk approaching expiration")
        elif days_to_exp <= 10:
            urgency += 0.5
            exit_type = "time"
            reasons.append("Assignment risk increasing")

        # Profit target exit
        current_pnl = position.calculate_unrealized_pnl()
        max_profit = position.calculate_max_profit()

        if max_profit > 0:
            profit_percent = current_pnl / max_profit
            if profit_percent >= 0.6:  # 60% of max profit
                urgency += 0.7
                exit_type = "profit"
                reasons.append(f"Good profit target reached: {profit_percent:.1%} of premium")
            elif profit_percent >= 0.4:  # 40% profit
                urgency += 0.4
                exit_type = "profit"
                reasons.append(f"Decent profit from theta decay: {profit_percent:.1%}")

        # Loss limit management
        if max_profit > 0:
            loss_ratio = abs(current_pnl) / max_profit
            if current_pnl < 0 and loss_ratio >= 1.5:  # Lost 150% of premium
                urgency = 1.0
                exit_type = "loss"
                reasons.append("Severe loss - close immediately")
            elif current_pnl < 0 and loss_ratio >= 0.75:  # Lost 75% of premium
                urgency += 0.7
                exit_type = "loss"
                reasons.append("Significant loss developing")

        # Volatility expansion risk
        if conditions.volatility_rank > 0.9:
            urgency += 0.5
            if not exit_type:
                exit_type = "adjustment"
            reasons.append("Extreme volatility expansion threatening position")

        # Regime change
        unfavorable_regimes = [8, 1, 2]  # Crisis, strong trends
        if conditions.regime in unfavorable_regimes:
            urgency += 0.4
            if not exit_type:
                exit_type = "adjustment"
            reasons.append(f"Regime {conditions.regime} dangerous for short strangle")

        # Price movement analysis (key for unlimited risk)
        if len(position.strikes) >= 2:
            put_strike = min(position.strikes)
            call_strike = max(position.strikes)
            current_price = position.current_underlying_price

            # Check if approaching either strike
            distance_to_put = abs(current_price - put_strike) / put_strike
            distance_to_call = abs(current_price - call_strike) / call_strike

            if current_price <= put_strike or current_price >= call_strike:
                urgency += 0.8
                exit_type = "loss"
                reasons.append("Price beyond strikes - high loss risk")
            elif distance_to_put < 0.05 or distance_to_call < 0.05:  # Within 5%
                urgency += 0.6
                if not exit_type:
                    exit_type = "adjustment"
                reasons.append("Price approaching strike boundaries")

        # Strong trend development
        if abs(conditions.trend_strength) > 0.7:
            urgency += 0.5
            if not exit_type:
                exit_type = "adjustment"
            reasons.append("Strong trend dangerous for short strangle")

        return ExitSignal(
            should_exit=urgency >= 0.5,  # Aggressive exit for unlimited risk
            urgency=min(urgency, 1.0),
            exit_type=exit_type or "hold",
            reasons=reasons or ["Position within acceptable parameters"]
        )

    def get_risk_metrics(self, strikes: List[int], underlying_price: int,
                        time_to_expiration: int, volatility: float) -> RiskMetrics:
        """
        Calculate risk metrics for Short Strangle.

        Args:
            strikes: [put_strike, call_strike] in cents (put_strike < call_strike)
            underlying_price: Current underlying price in cents
            time_to_expiration: Days to expiration
            volatility: Implied volatility

        Returns:
            RiskMetrics for the Short Strangle
        """
        if len(strikes) != 2:
            raise ValueError("Short Strangle requires exactly 2 strikes: [put_strike, call_strike]")

        put_strike, call_strike = strikes

        if put_strike >= call_strike:
            raise ValueError("Put strike must be lower than call strike")

        # Estimate premium collected from OTM options
        time_factor = min(time_to_expiration / 30, 2.0)
        vol_factor = volatility * 1.8

        # OTM options provide less premium
        put_otm_factor = max(0.3, 1.0 - (underlying_price - put_strike) / underlying_price)
        call_otm_factor = max(0.3, 1.0 - (call_strike - underlying_price) / underlying_price)

        put_premium = int(underlying_price * 0.02 * vol_factor * time_factor * put_otm_factor)
        call_premium = int(underlying_price * 0.02 * vol_factor * time_factor * call_otm_factor)
        total_premium = put_premium + call_premium

        # Risk calculations
        max_profit = total_premium  # Limited to premium collected
        max_loss = 2147483647  # Unlimited loss

        # Breakeven points
        breakeven_lower = put_strike - total_premium
        breakeven_upper = call_strike + total_premium
        breakevens = [breakeven_lower, breakeven_upper]

        # Probability estimation (higher than short straddle due to wider zone)
        profit_zone_width = call_strike - put_strike
        total_breakeven_width = breakeven_upper - breakeven_lower

        # Wider zone increases probability
        base_prob = max(0.3, 0.7 - volatility * 0.3)
        width_bonus = min(0.2, profit_zone_width / underlying_price * 2)  # Wider zone helps
        time_prob = max(0.1, 0.25 - time_to_expiration / 150)
        profit_probability = min(base_prob + width_bonus + time_prob, 0.7)

        # Risk/reward ratio
        expected_loss = total_premium * 2.5  # Conservative estimate
        risk_reward_ratio = max_profit / expected_loss if expected_loss > 0 else 0.4

        # Margin requirements (lower than short straddle due to OTM strikes)
        margin_per_strike = underlying_price * 15 // 100  # 15% for OTM options
        capital_requirement = margin_per_strike * 2  # For both strikes
        margin_requirement = int(capital_requirement * 1.4)  # 40% buffer

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
        Get position legs for Short Strangle construction.

        Args:
            strikes: [put_strike, call_strike] in cents (put_strike < call_strike)
            expiration_date: Option expiration date

        Returns:
            List of PositionLeg objects for Short Strangle
        """
        if len(strikes) != 2:
            raise ValueError("Short Strangle requires exactly 2 strikes: [put_strike, call_strike]")

        put_strike, call_strike = strikes

        return [
            # Short put at lower strike (OTM)
            PositionLeg(
                option_type=OptionType.PUT,
                strike=put_strike,
                quantity=-1,  # Short position
                expiration_date=expiration_date
            ),
            # Short call at higher strike (OTM)
            PositionLeg(
                option_type=OptionType.CALL,
                strike=call_strike,
                quantity=-1,  # Short position
                expiration_date=expiration_date
            )
        ]

    def _validate_strategy_strikes(self, strikes: List[int], underlying_price: int) -> bool:
        """
        Validate Short Strangle strike configuration.

        Args:
            strikes: Strike prices to validate
            underlying_price: Current underlying price

        Returns:
            True if strikes are valid for Short Strangle
        """
        if len(strikes) != 2:
            return False

        put_strike, call_strike = strikes

        # Put strike must be lower than call strike
        if put_strike >= call_strike:
            return False

        # Current price should be between the strikes
        if not (put_strike < underlying_price < call_strike):
            return False

        # Strikes should be reasonably OTM (3% to 25% OTM for short positions)
        put_distance = (underlying_price - put_strike) / underlying_price
        call_distance = (call_strike - underlying_price) / underlying_price

        if put_distance < 0.03 or put_distance > 0.25:
            return False

        if call_distance < 0.03 or call_distance > 0.25:
            return False

        # Don't require as tight balance as long strangle
        if abs(put_distance - call_distance) > 0.15:  # Allow 15% difference
            return False

        return True

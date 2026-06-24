"""
Position state data models for options trading strategies.

Provides data structures for representing options positions with strategy types,
price zones, and comprehensive position tracking.
"""

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import List, Optional
from datetime import datetime


class StrategyType(Enum):
    """Enumeration of the 16 core options strategies supported by the system."""

    # Single leg strategies
    LONG_CALL = "long_call"
    SHORT_CALL = "short_call"
    LONG_PUT = "long_put"
    SHORT_PUT = "short_put"

    # Vertical spreads
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_CALL_SPREAD = "bear_call_spread"
    BULL_PUT_SPREAD = "bull_put_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"

    # Horizontal spreads
    CALENDAR_CALL = "calendar_call"
    CALENDAR_PUT = "calendar_put"

    # Volatility strategies
    LONG_STRADDLE = "long_straddle"
    SHORT_STRADDLE = "short_straddle"
    LONG_STRANGLE = "long_strangle"
    SHORT_STRANGLE = "short_strangle"

    # Complex strategies
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"


class PositionZones(IntEnum):
    """
    Position profit/loss zones for spatial representation.

    Maps position P/L to discrete zones for neural network processing.
    """
    DEEP_LOSS = -3      # < -80% of max loss
    LOSS = -2           # -50% to -80% of max loss
    WARNING = -1        # -20% to -50% of max loss
    SAFE = 0            # -20% to +20% of max profit
    PROFIT = 1          # +20% to +50% of max profit
    HIGH_PROFIT = 2     # +50% to +80% of max profit
    MAX_PROFIT = 3      # > +80% of max profit


class OptionType(Enum):
    """Option type enumeration."""
    CALL = "CALL"
    PUT = "PUT"


@dataclass(init=False)
class Position:
    """
    Comprehensive position data model for options strategies.

    Stores all necessary information for position tracking, Greeks calculation,
    and P/L analysis. All monetary values stored in cents to avoid float precision issues.
    """

    # Strategy identification
    strategy_type: StrategyType

    # Timing
    entry_date: datetime
    expiration_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None

    # Position structure
    strikes: List[int]                    # Strike prices in cents
    option_types: List[OptionType]        # CALL or PUT for each leg
    quantities: List[int]                 # Number of contracts (negative for short)

    # Pricing
    entry_prices: List[int]               # Entry prices per contract in cents
    current_prices: List[int]             # Current market prices in cents
    underlying_price_at_entry: int        # Underlying price at entry in cents
    current_underlying_price: int         # Current underlying price in cents

    # Optional tracking
    adjustments_made: int = 0             # Number of adjustments made to position

    def __init__(
        self,
        strategy_type: StrategyType,
        entry_date: datetime,
        expiration_date: Optional[datetime] = None,
        expiry_date: Optional[datetime] = None,
        strikes: Optional[List[int]] = None,
        option_types: Optional[List[OptionType]] = None,
        quantities: Optional[List[int]] = None,
        entry_prices: Optional[List[int]] = None,
        current_prices: Optional[List[int]] = None,
        underlying_price_at_entry: Optional[int] = None,
        current_underlying_price: Optional[int] = None,
        adjustments_made: int = 0
    ):
        self.strategy_type = strategy_type
        self.entry_date = entry_date
        self.expiration_date = expiration_date
        self.expiry_date = expiry_date
        self.strikes = strikes or []
        self.option_types = option_types or []
        self.quantities = quantities or []
        self.entry_prices = entry_prices or []
        self.current_prices = current_prices or []
        self.underlying_price_at_entry = underlying_price_at_entry or 0
        self.current_underlying_price = current_underlying_price or 0
        self.adjustments_made = adjustments_made

        self.__post_init__()

    def __post_init__(self):
        """Validate position data consistency."""
        if self.expiration_date is None and self.expiry_date is not None:
            self.expiration_date = self.expiry_date

        if isinstance(self.entry_date, str):
            self.entry_date = datetime.fromisoformat(self.entry_date)
        if isinstance(self.expiration_date, str):
            self.expiration_date = datetime.fromisoformat(self.expiration_date)

        if self.expiration_date is None:
            raise ValueError("expiration_date must be provided")

        n_legs = len(self.strikes)

        if not all(len(lst) == n_legs for lst in [
            self.option_types, self.quantities, self.entry_prices, self.current_prices
        ]):
            raise ValueError("All position arrays must have same length")

        if n_legs == 0:
            raise ValueError("Position must have at least one leg")

    @property
    def days_to_expiration(self) -> int:
        """Calculate days to expiration from current date."""
        return (self.expiration_date - datetime.now()).days

    @property
    def days_held(self) -> int:
        """Calculate number of days position has been held."""
        return (datetime.now() - self.entry_date).days

    def calculate_price_zone(self) -> PositionZones:
        """
        Calculate current price zone based on P/L relative to max profit/loss.

        Returns:
            PositionZones enum value representing current position state
        """
        unrealized_pnl = self.calculate_unrealized_pnl()
        max_profit = self.calculate_max_profit()
        max_loss = self.calculate_max_loss()

        if max_profit == 0 and max_loss == 0:
            return PositionZones.SAFE

        if unrealized_pnl >= 0:
            # Profit territory
            if max_profit == 0:
                return PositionZones.SAFE

            pnl_ratio = unrealized_pnl / max_profit
            if pnl_ratio > 0.8:
                return PositionZones.MAX_PROFIT
            elif pnl_ratio > 0.5:
                return PositionZones.HIGH_PROFIT
            elif pnl_ratio > 0.2:
                return PositionZones.PROFIT
            else:
                return PositionZones.SAFE
        else:
            # Loss territory
            if max_loss == 0:
                return PositionZones.SAFE

            loss_ratio = abs(unrealized_pnl) / abs(max_loss)
            if loss_ratio > 0.8:
                return PositionZones.DEEP_LOSS
            elif loss_ratio > 0.5:
                return PositionZones.LOSS
            elif loss_ratio > 0.2:
                return PositionZones.WARNING
            else:
                return PositionZones.SAFE

    def calculate_unrealized_pnl(self) -> int:
        """
        Calculate current unrealized P/L in cents.

        Returns:
            Unrealized P/L in cents (positive = profit, negative = loss)
        """
        current_value = sum(
            quantity * current_price
            for quantity, current_price in zip(self.quantities, self.current_prices)
        )

        entry_value = sum(
            quantity * entry_price
            for quantity, entry_price in zip(self.quantities, self.entry_prices)
        )

        return current_value - entry_value

    def calculate_max_profit(self) -> int:
        """
        Calculate theoretical maximum profit for the strategy.

        Returns:
            Maximum profit in cents (0 for unlimited profit strategies)
        """
        if self.strategy_type == StrategyType.LONG_CALL:
            return 0  # Unlimited
        elif self.strategy_type == StrategyType.SHORT_CALL:
            return self.entry_prices[0] * self.quantities[0]
        elif self.strategy_type == StrategyType.LONG_PUT:
            return (self.strikes[0] - self.entry_prices[0]) * abs(self.quantities[0])
        elif self.strategy_type == StrategyType.SHORT_PUT:
            return self.entry_prices[0] * abs(self.quantities[0])

        # For spreads and complex strategies, calculate based on strike differences
        elif self.strategy_type in [StrategyType.BULL_CALL_SPREAD, StrategyType.BEAR_PUT_SPREAD]:
            # Debit spread: max profit = spread width - net debit paid. The old
            # max(width - debit, debit) fallback returned the debit (i.e. the
            # max loss) whenever the debit exceeded half the width, which is
            # never the correct max profit.
            spread_width = abs(self.strikes[1] - self.strikes[0])
            net_debit = abs(sum(
                qty * price for qty, price in zip(self.quantities, self.entry_prices)
            ))
            return spread_width * abs(self.quantities[0]) - net_debit

        elif self.strategy_type in [StrategyType.BEAR_CALL_SPREAD, StrategyType.BULL_PUT_SPREAD]:
            net_credit = sum(
                -qty * price for qty, price in zip(self.quantities, self.entry_prices)
            )
            return abs(net_credit)

        elif self.strategy_type == StrategyType.IRON_CONDOR:
            # Maximum profit is the net credit received
            net_credit = sum(
                -qty * price for qty, price in zip(self.quantities, self.entry_prices)
            )
            return abs(net_credit)

        else:
            # For other strategies, use a simplified calculation
            return abs(sum(self.entry_prices[i] * self.quantities[i] for i in range(len(self.strikes))))

    def calculate_max_loss(self) -> int:
        """
        Calculate theoretical maximum loss for the strategy.

        Returns:
            Maximum loss in cents (0 for unlimited loss strategies, returned as positive)
        """
        if self.strategy_type == StrategyType.SHORT_CALL:
            return 0  # Unlimited
        elif self.strategy_type == StrategyType.LONG_CALL:
            return abs(self.entry_prices[0] * self.quantities[0])
        elif self.strategy_type == StrategyType.LONG_PUT:
            return abs(self.entry_prices[0] * self.quantities[0])
        elif self.strategy_type == StrategyType.SHORT_PUT:
            return abs((self.strikes[0] - self.entry_prices[0]) * self.quantities[0])

        # For spreads
        elif self.strategy_type in [StrategyType.BULL_CALL_SPREAD, StrategyType.BEAR_PUT_SPREAD]:
            spread_width = abs(self.strikes[1] - self.strikes[0])
            net_debit = sum(
                qty * price for qty, price in zip(self.quantities, self.entry_prices)
            )
            return abs(net_debit)

        elif self.strategy_type in [StrategyType.BEAR_CALL_SPREAD, StrategyType.BULL_PUT_SPREAD]:
            spread_width = abs(self.strikes[1] - self.strikes[0])
            net_credit = sum(
                -qty * price for qty, price in zip(self.quantities, self.entry_prices)
            )
            return abs(spread_width * abs(self.quantities[0]) - abs(net_credit))

        elif self.strategy_type == StrategyType.IRON_CONDOR:
            # Maximum loss is spread width minus net credit
            if len(self.strikes) < 4:
                net_credit = sum(
                    -qty * price for qty, price in zip(self.quantities, self.entry_prices)
                )
                return abs(net_credit)

            put_spread_width = abs(self.strikes[1] - self.strikes[0])  # Assuming sorted strikes
            call_spread_width = abs(self.strikes[3] - self.strikes[2])
            spread_width = max(put_spread_width, call_spread_width)
            net_credit = sum(
                -qty * price for qty, price in zip(self.quantities, self.entry_prices)
            )
            return abs(spread_width * abs(self.quantities[0]) - abs(net_credit))

        else:
            # For other strategies, use entry cost as max loss
            total_cost = sum(
                qty * price for qty, price in zip(self.quantities, self.entry_prices)
                if qty > 0  # Only count long positions
            )
            return abs(total_cost)

    def calculate_breakevens(self) -> List[int]:
        """
        Calculate breakeven points for the strategy.

        Returns:
            List of breakeven prices in cents
        """
        breakevens = []

        if self.strategy_type == StrategyType.LONG_CALL:
            breakevens.append(self.strikes[0] + self.entry_prices[0])

        elif self.strategy_type == StrategyType.SHORT_CALL:
            breakevens.append(self.strikes[0] + self.entry_prices[0])

        elif self.strategy_type == StrategyType.LONG_PUT:
            breakevens.append(self.strikes[0] - self.entry_prices[0])

        elif self.strategy_type == StrategyType.SHORT_PUT:
            breakevens.append(self.strikes[0] - self.entry_prices[0])

        elif self.strategy_type in [
            StrategyType.BULL_CALL_SPREAD,
            StrategyType.BEAR_CALL_SPREAD,
            StrategyType.BULL_PUT_SPREAD,
            StrategyType.BEAR_PUT_SPREAD
        ]:
            # For spreads, calculate net cost/credit and adjust strike
            net_cost = sum(
                qty * price for qty, price in zip(self.quantities, self.entry_prices)
            )

            if self.strategy_type in [StrategyType.BULL_CALL_SPREAD, StrategyType.BEAR_PUT_SPREAD]:
                breakevens.append(min(self.strikes) + abs(net_cost))
            else:
                breakevens.append(max(self.strikes) - abs(net_cost))

        elif self.strategy_type == StrategyType.IRON_CONDOR:
            # Iron condor has two breakevens
            net_credit = sum(
                -qty * price for qty, price in zip(self.quantities, self.entry_prices)
            )
            sorted_strikes = sorted(self.strikes)
            breakevens.append(sorted_strikes[1] - abs(net_credit))  # Put side
            breakevens.append(sorted_strikes[2] + abs(net_credit))  # Call side

        elif self.strategy_type in [StrategyType.LONG_STRADDLE, StrategyType.SHORT_STRADDLE]:
            net_cost = sum(
                qty * price for qty, price in zip(self.quantities, self.entry_prices)
            )
            strike = self.strikes[0]  # Assuming same strike for straddle
            breakevens.append(strike - abs(net_cost))
            breakevens.append(strike + abs(net_cost))

        elif self.strategy_type in [StrategyType.LONG_STRANGLE, StrategyType.SHORT_STRANGLE]:
            net_cost = sum(
                qty * price for qty, price in zip(self.quantities, self.entry_prices)
            )
            put_strike = min(self.strikes)
            call_strike = max(self.strikes)
            breakevens.append(put_strike - abs(net_cost))
            breakevens.append(call_strike + abs(net_cost))

        else:
            # Default for other strategies
            breakevens.append(self.strikes[0])

        return breakevens

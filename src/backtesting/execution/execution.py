"""
Order execution handler with realistic trading costs

Simulates order fills with slippage and commission costs
appropriate for options trading.
"""

from datetime import datetime
from typing import Optional, Dict
import logging

from ..core.events import OrderEvent, FillEvent


class ExecutionHandler:
    """
    Order execution simulator with realistic trading costs

    Applies slippage and commission costs to simulate real trading conditions.
    Slippage rates are calibrated for options trading (typically 0.05-0.1%).
    """

    def __init__(self, slippage_rate: float = 0.0005, commission_rate: float = 1.0):
        """
        Initialize execution handler with cost parameters

        Args:
            slippage_rate: Slippage as percentage of trade value (default 0.05% for options)
            commission_rate: Commission per contract (default $1.00)
        """
        self.slippage_rate = slippage_rate
        self.commission_rate = commission_rate

        # Latest market prices for execution
        self.latest_prices: Dict[str, float] = {}

        # Execution statistics
        self.orders_executed = 0
        self.total_slippage = 0.0
        self.total_commission = 0.0

        self.logger = logging.getLogger(__name__)

    def update_prices(self, symbol: str, price: float):
        """
        Update latest market price for a symbol

        Args:
            symbol: Trading symbol
            price: Current market price
        """
        self.latest_prices[symbol] = price

    def execute_order(self, order: OrderEvent) -> Optional[FillEvent]:
        """
        Execute order with realistic slippage and commission

        Args:
            order: OrderEvent to execute

        Returns:
            FillEvent with execution details or None if cannot execute
        """
        symbol = order.symbol

        # Check if we have market price for this symbol
        if symbol not in self.latest_prices:
            self.logger.warning(f"No market price available for {symbol}")
            return None

        market_price = self.latest_prices[symbol]
        quantity = order.quantity
        direction = order.direction

        # Calculate slippage
        if direction == 'BUY':
            # BUY orders experience positive slippage (pay more)
            fill_price = market_price * (1 + self.slippage_rate)
        elif direction == 'SELL':
            # SELL orders experience negative slippage (receive less)
            fill_price = market_price * (1 - self.slippage_rate)
        else:
            self.logger.error(f"Unknown order direction: {direction}")
            return None

        # Calculate trade value (excluding commission)
        trade_value = fill_price * quantity

        # Calculate commission
        commission = self.commission_rate * quantity

        # Total fill cost (includes commission for BUY, excludes for SELL)
        if direction == 'BUY':
            fill_cost = trade_value + commission
        else:
            fill_cost = trade_value - commission

        # Update execution statistics
        self.orders_executed += 1
        slippage_cost = abs(fill_price - market_price) * quantity
        self.total_slippage += slippage_cost
        self.total_commission += commission

        # Create fill event
        fill_event = FillEvent(
            timestamp=order.timestamp,
            symbol=symbol,
            order_type=order.order_type,
            quantity=quantity,
            direction=direction,
            fill_cost=fill_cost,
            commission=commission
        )

        self.logger.debug(
            f"Executed {direction} {quantity} {symbol} at {fill_price:.2f} "
            f"(market: {market_price:.2f}, slippage: {slippage_cost:.2f}, "
            f"commission: {commission:.2f})"
        )

        return fill_event

    def get_execution_stats(self) -> Dict[str, float]:
        """Get execution statistics summary"""
        return {
            'orders_executed': self.orders_executed,
            'total_slippage': self.total_slippage,
            'total_commission': self.total_commission,
            'avg_slippage_per_order': (
                self.total_slippage / self.orders_executed
                if self.orders_executed > 0 else 0.0
            )
        }
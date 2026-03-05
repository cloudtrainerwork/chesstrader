"""
Portfolio state management for backtesting

Tracks positions, cash, commissions, and portfolio value over time.
Provides mark-to-market updates and equity curve generation.
"""

import copy
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

from ..core.events import MarketEvent, FillEvent, SignalEvent, OrderEvent


class Portfolio:
    """
    Portfolio state manager for backtesting

    Tracks current positions, cash balances, and portfolio value.
    Maintains time series of all holdings for equity curve analysis.
    """

    def __init__(self, start_date: datetime, initial_cash: float = 100000.0):
        """
        Initialize portfolio with starting cash

        Args:
            start_date: Backtest start date
            initial_cash: Starting cash amount
        """
        self.start_date = start_date
        self.initial_cash = initial_cash

        # Current position tracking (symbol -> quantity)
        self.current_positions = defaultdict(int)

        # Current holdings (cash, commission, total, positions)
        self.current_holdings = {
            'cash': initial_cash,
            'commission': 0.0,
            'total': initial_cash
        }

        # Time series of all holdings for equity curve
        self.all_holdings = []
        self._record_holdings(start_date)

        # Position cost basis tracking for P&L calculation
        self.position_costs = defaultdict(float)  # symbol -> total cost
        self.position_shares = defaultdict(int)   # symbol -> total shares

        # Latest market prices for mark-to-market
        self.latest_prices = {}

    def _record_holdings(self, timestamp: datetime):
        """Record current holdings state with timestamp"""
        holdings_record = copy.deepcopy(self.current_holdings)
        holdings_record['datetime'] = timestamp

        # Add individual position values
        for symbol in self.current_positions:
            if symbol in self.latest_prices and self.current_positions[symbol] != 0:
                market_value = self.current_positions[symbol] * self.latest_prices[symbol]
                holdings_record[symbol] = market_value

        self.all_holdings.append(holdings_record)

    def update_market_value(self, event: MarketEvent):
        """
        Update portfolio mark-to-market value on new market data

        Args:
            event: MarketEvent with new price information
        """
        # Store latest price for mark-to-market calculation
        symbol = event.symbol
        if 'price' in event.data:
            self.latest_prices[symbol] = event.data['price']
        elif 'close' in event.data:
            self.latest_prices[symbol] = event.data['close']

        # Recalculate total portfolio value
        self._calculate_total_value()

        # Record new holdings state
        self._record_holdings(event.timestamp)

    def _calculate_total_value(self):
        """Calculate total portfolio value including positions and cash"""
        position_value = 0.0

        for symbol, quantity in self.current_positions.items():
            if symbol in self.latest_prices and quantity != 0:
                position_value += quantity * self.latest_prices[symbol]

        self.current_holdings['total'] = (
            self.current_holdings['cash'] +
            position_value -
            self.current_holdings['commission']
        )

    def update_fill(self, event: FillEvent):
        """
        Update portfolio positions and cash based on fill event

        Args:
            event: FillEvent with trade execution details
        """
        symbol = event.symbol
        quantity = event.quantity
        direction = event.direction
        fill_cost = event.fill_cost
        commission = event.commission

        # Update position quantity
        if direction == 'BUY':
            self.current_positions[symbol] += quantity
            self.current_holdings['cash'] -= fill_cost
        elif direction == 'SELL':
            self.current_positions[symbol] -= quantity
            self.current_holdings['cash'] += fill_cost

        # Track commission costs
        self.current_holdings['commission'] += commission

        # Update cost basis tracking
        if direction == 'BUY':
            # Add to position cost (excluding commission for simplicity)
            cost_without_commission = fill_cost - commission
            self.position_costs[symbol] += cost_without_commission
            self.position_shares[symbol] += quantity
        elif direction == 'SELL':
            # Reduce position cost proportionally
            if self.position_shares[symbol] > 0:
                cost_per_share = self.position_costs[symbol] / self.position_shares[symbol]
                self.position_costs[symbol] -= cost_per_share * quantity
                self.position_shares[symbol] -= quantity

        # Recalculate total portfolio value
        self._calculate_total_value()

    def generate_orders(self, event: SignalEvent) -> List[OrderEvent]:
        """
        Generate order events from strategy signals

        Args:
            event: SignalEvent from strategy

        Returns:
            List of OrderEvent objects
        """
        orders = []

        # Simple implementation: convert signals to market orders
        symbol = event.symbol
        signal_type = event.signal_type
        strategy_id = event.strategy_id

        # Basic position sizing: 1% of portfolio value
        portfolio_value = self.current_holdings['total']
        position_value = portfolio_value * 0.01  # 1% allocation

        if symbol in self.latest_prices:
            price = self.latest_prices[symbol]
            quantity = int(position_value / price)

            if quantity > 0:
                if signal_type == 'LONG':
                    # Only buy if we don't already have a position
                    if self.current_positions[symbol] <= 0:
                        order = OrderEvent(
                            timestamp=event.timestamp,
                            symbol=symbol,
                            order_type='MKT',
                            quantity=quantity,
                            direction='BUY'
                        )
                        orders.append(order)

                elif signal_type == 'SHORT':
                    # Sell existing position or go short
                    if self.current_positions[symbol] > 0:
                        # Close long position first
                        close_quantity = min(quantity, self.current_positions[symbol])
                        order = OrderEvent(
                            timestamp=event.timestamp,
                            symbol=symbol,
                            order_type='MKT',
                            quantity=close_quantity,
                            direction='SELL'
                        )
                        orders.append(order)

        return orders

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Generate portfolio performance summary

        Returns:
            Dictionary with performance metrics
        """
        if len(self.all_holdings) < 2:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0
            }

        # Extract equity curve values
        equity_values = [holding['total'] for holding in self.all_holdings]
        equity_array = np.array(equity_values)

        # Calculate total return
        total_return = (equity_values[-1] - equity_values[0]) / equity_values[0]

        # Calculate daily returns
        daily_returns = np.diff(equity_array) / equity_array[:-1]

        # Calculate Sharpe ratio (assuming 252 trading days)
        sharpe_ratio = 0.0
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            annualized_return = np.mean(daily_returns) * 252
            annualized_volatility = np.std(daily_returns) * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_volatility

        # Calculate maximum drawdown
        running_max = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - running_max) / running_max
        max_drawdown = np.min(drawdowns)

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': equity_values[-1],
            'initial_value': equity_values[0],
            'total_trades': len([h for h in self.all_holdings if 'commission' in h])
        }

    def initialize(self, start_date: datetime):
        """Initialize portfolio for backtest start"""
        # Reset state if needed
        pass

    def get_equity_curve(self) -> List[Dict]:
        """Return complete equity curve for analysis"""
        return self.all_holdings
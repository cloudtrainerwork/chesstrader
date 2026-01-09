"""
Position P/L calculation and analysis.

Provides comprehensive P/L tracking, percentage calculations,
and position performance metrics.
"""

from typing import Dict, List
from datetime import datetime
import math

from .position_models import Position


class PnLCalculator:
    """
    Comprehensive P/L calculator for options positions.

    Calculates unrealized P/L, percentages, and position metrics
    for risk management and performance analysis.
    """

    def __init__(self):
        """Initialize P/L calculator."""
        pass

    def unrealized_pnl(self, position: Position, current_prices: List[int]) -> int:
        """
        Calculate unrealized P/L for a position.

        Args:
            position: Position object
            current_prices: Current market prices for each leg (in cents)

        Returns:
            Unrealized P/L in cents (positive = profit, negative = loss)
        """
        if len(current_prices) != len(position.quantities):
            raise ValueError("Current prices must match position legs")

        # Calculate current market value
        current_value = sum(
            quantity * current_price
            for quantity, current_price in zip(position.quantities, current_prices)
        )

        # Calculate entry value
        entry_value = sum(
            quantity * entry_price
            for quantity, entry_price in zip(position.quantities, position.entry_prices)
        )

        return current_value - entry_value

    def pnl_percentage(self, position: Position, current_prices: List[int]) -> float:
        """
        Calculate P/L as percentage of initial investment.

        Args:
            position: Position object
            current_prices: Current market prices for each leg

        Returns:
            P/L percentage (1.0 = 100% profit, -1.0 = 100% loss)
        """
        unrealized = self.unrealized_pnl(position, current_prices)

        # Calculate total cost of long positions (initial investment)
        long_cost = sum(
            quantity * entry_price
            for quantity, entry_price in zip(position.quantities, position.entry_prices)
            if quantity > 0
        )

        if long_cost == 0:
            # For credit spreads, use credit received as base
            credit_received = sum(
                -quantity * entry_price
                for quantity, entry_price in zip(position.quantities, position.entry_prices)
                if quantity < 0
            )
            if credit_received == 0:
                return 0.0
            return float(unrealized / credit_received)

        return float(unrealized / long_cost)

    def percent_of_max_profit(self, position: Position, current_prices: List[int]) -> float:
        """
        Calculate P/L as percentage of maximum possible profit.

        Args:
            position: Position object
            current_prices: Current market prices for each leg

        Returns:
            Percentage of max profit (1.0 = 100% of max profit achieved)
        """
        unrealized = self.unrealized_pnl(position, current_prices)
        max_profit = position.calculate_max_profit()

        if max_profit == 0:
            # Unlimited profit strategies
            if unrealized <= 0:
                return 0.0
            else:
                # Normalize large profits to reasonable scale
                return min(float(unrealized / 1000), 3.0)  # Cap at 3x for normalization

        return float(unrealized / max_profit)

    def percent_of_max_loss(self, position: Position, current_prices: List[int]) -> float:
        """
        Calculate P/L as percentage of maximum possible loss.

        Args:
            position: Position object
            current_prices: Current market prices for each leg

        Returns:
            Percentage of max loss (1.0 = 100% of max loss realized)
        """
        unrealized = self.unrealized_pnl(position, current_prices)
        max_loss = position.calculate_max_loss()

        if max_loss == 0:
            # Unlimited loss strategies
            if unrealized >= 0:
                return 0.0
            else:
                # Normalize large losses to reasonable scale
                return min(float(abs(unrealized) / 1000), 3.0)  # Cap at 3x for normalization

        if unrealized >= 0:
            return 0.0

        return float(abs(unrealized) / max_loss)

    def days_held(self, position: Position) -> int:
        """
        Calculate number of days position has been held.

        Args:
            position: Position object

        Returns:
            Number of days held
        """
        return (datetime.now() - position.entry_date).days

    def calculate_pnl_metrics(self, position: Position, current_prices: List[int]) -> Dict[str, float]:
        """
        Calculate comprehensive P/L metrics for a position.

        Args:
            position: Position object
            current_prices: Current market prices for each leg

        Returns:
            Dictionary with all P/L metrics
        """
        unrealized = self.unrealized_pnl(position, current_prices)

        # Calculate entry credit/debit
        entry_credit = sum(
            -quantity * entry_price
            for quantity, entry_price in zip(position.quantities, position.entry_prices)
            if quantity < 0
        )

        current_value = sum(
            quantity * current_price
            for quantity, current_price in zip(position.quantities, current_prices)
        )

        metrics = {
            'entry_credit': float(entry_credit),
            'current_value': float(current_value),
            'unrealized_pnl': float(unrealized),
            'pnl_percentage': self.pnl_percentage(position, current_prices),
            'percent_of_max_profit': self.percent_of_max_profit(position, current_prices),
            'percent_of_max_loss': self.percent_of_max_loss(position, current_prices),
            'days_held': float(self.days_held(position))
        }

        return metrics

    def position_performance_summary(self, position: Position, current_prices: List[int]) -> Dict[str, any]:
        """
        Generate comprehensive position performance summary.

        Args:
            position: Position object
            current_prices: Current market prices for each leg

        Returns:
            Dictionary with performance summary
        """
        metrics = self.calculate_pnl_metrics(position, current_prices)

        # Calculate additional performance metrics
        max_profit = position.calculate_max_profit()
        max_loss = position.calculate_max_loss()
        breakevens = position.calculate_breakevens()

        # Risk metrics
        risk_reward_ratio = 0.0
        if max_loss > 0:
            risk_reward_ratio = max_profit / max_loss if max_profit > 0 else float('inf')

        # Time metrics
        days_to_exp = position.days_to_expiration
        percent_time_remaining = max(days_to_exp / 30.0, 0.0) if days_to_exp >= 0 else 0.0

        # Current zone assessment
        price_zone = position.calculate_price_zone()

        summary = {
            'pnl_metrics': metrics,
            'position_structure': {
                'strategy_type': position.strategy_type.value,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'breakevens': breakevens,
                'risk_reward_ratio': risk_reward_ratio
            },
            'timing': {
                'days_held': metrics['days_held'],
                'days_to_expiration': days_to_exp,
                'percent_time_remaining': percent_time_remaining,
                'theta_pressure': max(0.0, (30 - days_to_exp) / 30.0) if days_to_exp >= 0 else 1.0
            },
            'risk_assessment': {
                'current_zone': price_zone.value,
                'zone_name': price_zone.name,
                'distance_to_breakevens': self._calculate_breakeven_distances(position, breakevens)
            }
        }

        return summary

    def _calculate_breakeven_distances(self, position: Position, breakevens: List[int]) -> Dict[str, float]:
        """Calculate distances to breakeven points."""
        current_price = position.current_underlying_price

        if not breakevens:
            return {'lower': 0.0, 'upper': 0.0}

        distances = {
            'lower': float((current_price - min(breakevens)) / current_price),
            'upper': float((max(breakevens) - current_price) / current_price)
        }

        if len(breakevens) == 1:
            # Single breakeven
            distances['upper'] = distances['lower']

        return distances
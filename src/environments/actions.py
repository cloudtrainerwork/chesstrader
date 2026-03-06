"""
Action definitions for the options trading environment.

Defines the discrete actions an agent can take to manage options positions.
"""

from enum import IntEnum
from typing import Dict, Any, Optional, Tuple
import numpy as np

from ..strategies.base import StrategyType


class ActionType(IntEnum):
    """
    Discrete actions for position management.

    These map to the gym.spaces.Discrete(4) action space.
    """
    HOLD = 0  # Keep position unchanged
    CLOSE = 1  # Close entire position
    ADJUST = 2  # Modify strikes or quantities
    ROLL = 3  # Move to next expiration


class PositionAdjustment:
    """Handles position adjustment logic for ADJUST and ROLL actions."""

    @staticmethod
    def adjust_position(position_state, market_price: float) -> Dict[str, Any]:
        """
        Adjust position strikes or quantities.

        For now, implements simple adjustments:
        - Move strikes closer to current price if too far OTM
        - Reduce position size if risk is too high

        Args:
            position_state: Current position state
            market_price: Current market price

        Returns:
            Adjustment details dictionary
        """
        adjustment = {
            'type': 'adjust',
            'original_strikes': position_state.strikes.copy(),
            'new_strikes': position_state.strikes.copy(),
            'original_quantities': position_state.quantities.copy(),
            'new_quantities': position_state.quantities.copy(),
            'cost': 0.0
        }

        # Simple adjustment logic - move strikes if price moved significantly
        price_move = abs(market_price - position_state.entry_price) / position_state.entry_price

        if price_move > 0.05:  # 5% move
            # Adjust strikes proportionally
            strike_adjustment = (market_price - position_state.entry_price) * 0.5
            adjustment['new_strikes'] = position_state.strikes + strike_adjustment

            # Adjustment cost (simplified)
            adjustment['cost'] = abs(strike_adjustment) * 10  # $10 per point moved

        return adjustment

    @staticmethod
    def roll_position(position_state, days_to_add: int = 30) -> Dict[str, Any]:
        """
        Roll position to next expiration.

        Args:
            position_state: Current position state
            days_to_add: Days to add for new expiration

        Returns:
            Roll details dictionary
        """
        roll = {
            'type': 'roll',
            'original_expiry': position_state.days_to_expiry,
            'new_expiry': position_state.days_to_expiry + days_to_add,
            'cost': 0.0
        }

        # Rolling cost (simplified - based on time value)
        time_value_cost = days_to_add * 2  # $2 per day rolled
        roll['cost'] = time_value_cost

        return roll


class ActionValidator:
    """Validates whether actions are valid given current state."""

    @staticmethod
    def is_action_valid(action: ActionType, position_state, current_step: int,
                       max_steps: int) -> Tuple[bool, str]:
        """
        Check if an action is valid given current state.

        Args:
            action: Action to validate
            position_state: Current position state
            current_step: Current step in episode
            max_steps: Maximum steps in episode

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # HOLD is always valid
        if action == ActionType.HOLD:
            return True, ""

        # CLOSE is always valid if position exists
        if action == ActionType.CLOSE:
            if position_state is None:
                return False, "No position to close"
            return True, ""

        # ADJUST is valid if enough time remains
        if action == ActionType.ADJUST:
            if position_state is None:
                return False, "No position to adjust"
            if position_state.days_to_expiry < 7:
                return False, "Too close to expiration to adjust"
            return True, ""

        # ROLL is valid if close to expiration
        if action == ActionType.ROLL:
            if position_state is None:
                return False, "No position to roll"
            if position_state.days_to_expiry > 14:
                return False, "Too early to roll (>14 days remaining)"
            return True, ""

        return False, f"Unknown action: {action}"


def calculate_action_cost(action: ActionType, position_state=None) -> float:
    """
    Calculate the cost of executing an action.

    Args:
        action: Action to execute
        position_state: Current position state

    Returns:
        Cost of the action (negative value)
    """
    if action == ActionType.HOLD:
        return 0.0  # No cost to hold

    if action == ActionType.CLOSE:
        # Closing cost based on number of contracts
        if position_state:
            num_contracts = len(position_state.strikes)
            return -num_contracts * 1.0  # $1 per contract
        return -4.0  # Default closing cost

    if action == ActionType.ADJUST:
        return -10.0  # Flat adjustment fee

    if action == ActionType.ROLL:
        return -15.0  # Rolling is more expensive

    return 0.0

"""
Base reward calculator for reinforcement learning environment.

Implements P/L-based rewards with risk adjustment and drawdown penalties.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

from .actions import ActionType


class RewardCalculator(ABC):
    """
    Base class for calculating rewards in the options trading environment.

    Implements immediate P/L rewards with risk adjustment and drawdown penalties.
    """

    def __init__(self,
                 sharpe_scaling: bool = True,
                 drawdown_threshold: float = 0.10,
                 drawdown_penalty_factor: float = 0.5):
        """
        Initialize reward calculator.

        Args:
            sharpe_scaling: Apply Sharpe ratio scaling to rewards
            drawdown_threshold: Threshold for applying drawdown penalty (10% default)
            drawdown_penalty_factor: Penalty multiplier when in drawdown (0.5 default)
        """
        self.sharpe_scaling = sharpe_scaling
        self.drawdown_threshold = drawdown_threshold
        self.drawdown_penalty_factor = drawdown_penalty_factor

        # Track rolling statistics for Sharpe calculation
        self.returns_history = []
        self.max_history_size = 100

    def calculate_reward(self,
                        position_state: Dict[str, Any],
                        action: str,
                        market_data: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate reward for the given action and state.

        Args:
            position_state: Current position state dictionary
            action: Action taken (HOLD, CLOSE, ADJUST, ROLL)
            market_data: Optional market data for context

        Returns:
            Calculated reward value
        """
        # Convert string action to ActionType if needed
        if isinstance(action, str):
            action = ActionType[action]

        # Calculate base P/L reward
        base_reward = self._calculate_pnl_reward(position_state, action)

        # Apply risk adjustment if enabled
        if self.sharpe_scaling and len(self.returns_history) > 1:
            base_reward = self._apply_sharpe_scaling(base_reward)

        # Apply drawdown penalty if applicable
        if self._is_in_drawdown(position_state):
            base_reward *= self.drawdown_penalty_factor

        # Apply position size normalization
        base_reward = self._normalize_by_position_size(base_reward, position_state)

        # Add strategy-specific components (overridden in subclasses)
        strategy_reward = self._calculate_strategy_specific_reward(
            position_state, action, market_data
        )

        total_reward = base_reward + strategy_reward

        # Update returns history for Sharpe calculation
        self._update_returns_history(total_reward)

        return total_reward

    def _calculate_pnl_reward(self,
                              position_state: Dict[str, Any],
                              action: ActionType) -> float:
        """
        Calculate basic P/L-based reward.

        Args:
            position_state: Current position state
            action: Action taken

        Returns:
            P/L-based reward
        """
        pnl = position_state.get('unrealized_pnl', 0)

        if action == ActionType.CLOSE:
            # Realized P/L on close
            return pnl

        elif action == ActionType.HOLD:
            # Change in unrealized P/L
            prev_pnl = position_state.get('prev_pnl', 0)
            return pnl - prev_pnl

        elif action == ActionType.ADJUST:
            # P/L change minus adjustment cost
            prev_pnl = position_state.get('prev_pnl', 0)
            adjustment_cost = position_state.get('adjustment_cost', 10)
            return (pnl - prev_pnl) - adjustment_cost

        elif action == ActionType.ROLL:
            # P/L change minus rolling cost
            prev_pnl = position_state.get('prev_pnl', 0)
            roll_cost = position_state.get('roll_cost', 15)
            return (pnl - prev_pnl) - roll_cost

        return 0

    def _apply_sharpe_scaling(self, reward: float) -> float:
        """
        Scale reward by Sharpe ratio (reward/volatility).

        Args:
            reward: Raw reward value

        Returns:
            Sharpe-scaled reward
        """
        if len(self.returns_history) < 2:
            return reward

        # Calculate volatility of recent returns
        returns = np.array(self.returns_history[-self.max_history_size:])
        volatility = np.std(returns)

        if volatility > 0:
            # Scale by inverse volatility (higher vol = lower reward)
            sharpe_factor = 1.0 / (1.0 + volatility)
            return reward * sharpe_factor

        return reward

    def _is_in_drawdown(self, position_state: Dict[str, Any]) -> bool:
        """
        Check if position is in drawdown state.

        Args:
            position_state: Current position state

        Returns:
            True if in drawdown exceeding threshold
        """
        pnl = position_state.get('unrealized_pnl', 0)
        max_loss = position_state.get('max_loss', -1000)

        if max_loss != 0:
            loss_percentage = abs(pnl / max_loss)
            return loss_percentage > self.drawdown_threshold

        return False

    def _normalize_by_position_size(self,
                                   reward: float,
                                   position_state: Dict[str, Any]) -> float:
        """
        Normalize reward by position size to prevent magnitude issues.

        Args:
            reward: Raw reward value
            position_state: Current position state

        Returns:
            Normalized reward
        """
        # Get position size metrics
        max_profit = abs(position_state.get('max_profit', 1000))
        max_loss = abs(position_state.get('max_loss', 1000))

        # Use maximum risk as normalizer
        normalizer = max(max_profit, max_loss)

        if normalizer > 0:
            return reward / normalizer * 100  # Scale to percentage basis

        return reward

    def _update_returns_history(self, reward: float):
        """
        Update rolling returns history for Sharpe calculation.

        Args:
            reward: Latest reward value
        """
        self.returns_history.append(reward)

        # Maintain maximum history size
        if len(self.returns_history) > self.max_history_size:
            self.returns_history.pop(0)

    def _calculate_strategy_specific_reward(self,
                                           position_state: Dict[str, Any],
                                           action: ActionType,
                                           market_data: Optional[Dict[str, Any]]) -> float:
        """
        Calculate strategy-specific reward components.

        Override in subclasses for strategy-specific rewards.

        Args:
            position_state: Current position state
            action: Action taken
            market_data: Optional market data

        Returns:
            Strategy-specific reward component
        """
        return 0.0

    def reset(self):
        """Reset calculator state for new episode."""
        self.returns_history = []
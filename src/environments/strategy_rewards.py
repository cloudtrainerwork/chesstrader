"""
Strategy-specific reward calculators for different options strategy categories.

Implements specialized reward components for neutral, directional, and volatility strategies.
"""

from typing import Dict, Any, Optional
import numpy as np

from .rewards import RewardCalculator
from .actions import ActionType
from ..strategies.base import StrategyType


class NeutralStrategyReward(RewardCalculator):
    """
    Reward calculator for neutral strategies (Iron Condor, Butterfly, etc.).

    Adds theta decay bonus and breach penalties specific to range-bound strategies.
    """

    def __init__(self,
                 theta_bonus_weight: float = 0.2,
                 breach_penalty_weight: float = 0.3,
                 **kwargs):
        """
        Initialize neutral strategy reward calculator.

        Args:
            theta_bonus_weight: Weight for theta decay bonus
            breach_penalty_weight: Weight for strike breach penalty
            **kwargs: Arguments passed to base class
        """
        super().__init__(**kwargs)
        self.theta_bonus_weight = theta_bonus_weight
        self.breach_penalty_weight = breach_penalty_weight

    def _calculate_strategy_specific_reward(self,
                                           position_state: Dict[str, Any],
                                           action: ActionType,
                                           market_data: Optional[Dict[str, Any]]) -> float:
        """
        Calculate neutral strategy-specific rewards.

        Args:
            position_state: Current position state
            action: Action taken
            market_data: Optional market data

        Returns:
            Strategy-specific reward component
        """
        reward = 0.0

        # Add theta decay bonus for time passing in profit zone
        theta_bonus = self.calculate_theta_bonus(position_state)
        reward += theta_bonus * self.theta_bonus_weight

        # Add breach penalty if price moves outside strikes
        breach_penalty = self.calculate_breach_penalty(position_state)
        reward += breach_penalty * self.breach_penalty_weight

        return reward

    def calculate_theta_bonus(self, position_state: Dict[str, Any]) -> float:
        """
        Calculate theta decay bonus for neutral strategies.

        Rewards time decay when position is in profit zone.

        Args:
            position_state: Current position state

        Returns:
            Theta bonus value
        """
        # Check if in profit zone
        current_price = position_state.get('current_price', 0)
        strikes = position_state.get('strikes', [])

        if len(strikes) >= 2:
            # For Iron Condor/Butterfly, profit zone is between inner strikes
            lower_strike = strikes[1] if len(strikes) > 1 else strikes[0]
            upper_strike = strikes[-2] if len(strikes) > 2 else strikes[-1]

            if lower_strike <= current_price <= upper_strike:
                # In profit zone - reward based on time decay
                days_to_expiry = position_state.get('days_to_expiry', 30)
                initial_days = position_state.get('initial_days_to_expiry', 30)

                if initial_days > 0:
                    time_decay_pct = 1.0 - (days_to_expiry / initial_days)
                    # More bonus as we approach expiration
                    theta_bonus = time_decay_pct * 10.0  # Scale factor
                    return theta_bonus

        return 0.0

    def calculate_breach_penalty(self, position_state: Dict[str, Any]) -> float:
        """
        Calculate penalty for breaching profit zone boundaries.

        Args:
            position_state: Current position state

        Returns:
            Breach penalty value (negative)
        """
        current_price = position_state.get('current_price', 0)
        strikes = position_state.get('strikes', [])

        if len(strikes) >= 2:
            # Define profit zone boundaries
            lower_strike = strikes[1] if len(strikes) > 1 else strikes[0]
            upper_strike = strikes[-2] if len(strikes) > 2 else strikes[-1]

            # Calculate distance from profit zone
            if current_price < lower_strike:
                breach_distance = (lower_strike - current_price) / lower_strike
                return -breach_distance * 20.0  # Negative penalty

            elif current_price > upper_strike:
                breach_distance = (current_price - upper_strike) / upper_strike
                return -breach_distance * 20.0  # Negative penalty

        return 0.0


class DirectionalStrategyReward(RewardCalculator):
    """
    Reward calculator for directional strategies (spreads, single legs).

    Adds momentum alignment bonus and early exit penalties.
    """

    def __init__(self,
                 momentum_bonus_weight: float = 0.3,
                 early_exit_penalty_weight: float = 0.2,
                 **kwargs):
        """
        Initialize directional strategy reward calculator.

        Args:
            momentum_bonus_weight: Weight for momentum alignment bonus
            early_exit_penalty_weight: Weight for early exit penalty
            **kwargs: Arguments passed to base class
        """
        super().__init__(**kwargs)
        self.momentum_bonus_weight = momentum_bonus_weight
        self.early_exit_penalty_weight = early_exit_penalty_weight

    def _calculate_strategy_specific_reward(self,
                                           position_state: Dict[str, Any],
                                           action: ActionType,
                                           market_data: Optional[Dict[str, Any]]) -> float:
        """
        Calculate directional strategy-specific rewards.

        Args:
            position_state: Current position state
            action: Action taken
            market_data: Optional market data

        Returns:
            Strategy-specific reward component
        """
        reward = 0.0

        # Add momentum bonus for holding when price moves favorably
        if action == ActionType.HOLD:
            momentum_bonus = self.calculate_momentum_bonus(position_state)
            reward += momentum_bonus * self.momentum_bonus_weight

        # Add early exit penalty for closing profitable positions too early
        if action == ActionType.CLOSE:
            early_exit_penalty = self.calculate_early_exit_penalty(position_state)
            reward += early_exit_penalty * self.early_exit_penalty_weight

        return reward

    def calculate_momentum_bonus(self, position_state: Dict[str, Any]) -> float:
        """
        Calculate momentum alignment bonus.

        Rewards holding when price moves in favorable direction.

        Args:
            position_state: Current position state

        Returns:
            Momentum bonus value
        """
        current_price = position_state.get('current_price', 0)
        entry_price = position_state.get('entry_price', 0)
        strategy_type = position_state.get('strategy_type')

        if entry_price == 0:
            return 0.0

        price_move = (current_price - entry_price) / entry_price

        # Determine if bullish or bearish strategy
        is_bullish = self._is_bullish_strategy(strategy_type)

        if is_bullish and price_move > 0:
            # Bullish strategy with upward price movement
            return price_move * 50.0  # Scale factor

        elif not is_bullish and price_move < 0:
            # Bearish strategy with downward price movement
            return abs(price_move) * 50.0  # Scale factor

        return 0.0

    def calculate_early_exit_penalty(self, position_state: Dict[str, Any]) -> float:
        """
        Calculate penalty for exiting profitable positions too early.

        Args:
            position_state: Current position state

        Returns:
            Early exit penalty (negative if too early)
        """
        pnl = position_state.get('unrealized_pnl', 0)
        days_to_expiry = position_state.get('days_to_expiry', 30)
        initial_days = position_state.get('initial_days_to_expiry', 30)

        if pnl > 0 and initial_days > 0:
            # Position is profitable
            time_held_pct = 1.0 - (days_to_expiry / initial_days)

            if time_held_pct < 0.5:  # Held for less than 50% of time
                # Penalty for early exit
                penalty = -(0.5 - time_held_pct) * 10.0
                return penalty

        return 0.0

    def _is_bullish_strategy(self, strategy_type) -> bool:
        """
        Determine if strategy is bullish.

        Args:
            strategy_type: Type of strategy

        Returns:
            True if bullish, False if bearish
        """
        if strategy_type:
            bullish_types = [
                StrategyType.LONG_CALL,
                StrategyType.BULL_CALL_SPREAD,
                StrategyType.BULL_PUT_SPREAD,
                StrategyType.SHORT_PUT
            ]
            return strategy_type in bullish_types

        return True  # Default to bullish


class VolatilityStrategyReward(RewardCalculator):
    """
    Reward calculator for volatility strategies (straddles, strangles).

    Adds volatility capture rewards and time decay penalties.
    """

    def __init__(self,
                 vol_capture_weight: float = 0.35,
                 time_decay_penalty_weight: float = 0.25,
                 **kwargs):
        """
        Initialize volatility strategy reward calculator.

        Args:
            vol_capture_weight: Weight for volatility capture reward
            time_decay_penalty_weight: Weight for time decay penalty
            **kwargs: Arguments passed to base class
        """
        super().__init__(**kwargs)
        self.vol_capture_weight = vol_capture_weight
        self.time_decay_penalty_weight = time_decay_penalty_weight

    def _calculate_strategy_specific_reward(self,
                                           position_state: Dict[str, Any],
                                           action: ActionType,
                                           market_data: Optional[Dict[str, Any]]) -> float:
        """
        Calculate volatility strategy-specific rewards.

        Args:
            position_state: Current position state
            action: Action taken
            market_data: Optional market data

        Returns:
            Strategy-specific reward component
        """
        reward = 0.0

        # Add volatility capture reward
        vol_reward = self.calculate_volatility_capture(position_state)
        reward += vol_reward * self.vol_capture_weight

        # Add time decay penalty for long premium strategies
        if self._is_long_premium(position_state.get('strategy_type')):
            decay_penalty = self.calculate_time_decay_penalty(position_state)
            reward += decay_penalty * self.time_decay_penalty_weight

        return reward

    def calculate_volatility_capture(self, position_state: Dict[str, Any]) -> float:
        """
        Calculate reward for capturing volatility moves.

        Args:
            position_state: Current position state

        Returns:
            Volatility capture reward
        """
        current_iv = position_state.get('current_iv', 0.25)
        entry_iv = position_state.get('entry_iv', 0.25)
        strategy_type = position_state.get('strategy_type')

        if entry_iv == 0:
            return 0.0

        iv_change = (current_iv - entry_iv) / entry_iv

        # Determine if long or short volatility
        is_long_vol = self._is_long_volatility(strategy_type)

        if is_long_vol and iv_change > 0:
            # Long vol with IV increase
            return iv_change * 30.0  # Scale factor

        elif not is_long_vol and iv_change < 0:
            # Short vol with IV decrease
            return abs(iv_change) * 30.0  # Scale factor

        return 0.0

    def calculate_time_decay_penalty(self, position_state: Dict[str, Any]) -> float:
        """
        Calculate time decay penalty for long premium strategies.

        Args:
            position_state: Current position state

        Returns:
            Time decay penalty (negative)
        """
        days_to_expiry = position_state.get('days_to_expiry', 30)
        initial_days = position_state.get('initial_days_to_expiry', 30)

        if initial_days > 0:
            time_decay_pct = 1.0 - (days_to_expiry / initial_days)

            # Accelerating penalty as expiration approaches
            decay_penalty = -time_decay_pct ** 2 * 15.0  # Quadratic penalty

            return decay_penalty

        return 0.0

    def _is_long_volatility(self, strategy_type) -> bool:
        """
        Determine if strategy is long volatility.

        Args:
            strategy_type: Type of strategy

        Returns:
            True if long volatility
        """
        if strategy_type:
            long_vol_types = [
                StrategyType.LONG_STRADDLE,
                StrategyType.LONG_STRANGLE,
                StrategyType.LONG_CALL,
                StrategyType.LONG_PUT
            ]
            return strategy_type in long_vol_types

        return True  # Default to long vol

    def _is_long_premium(self, strategy_type) -> bool:
        """
        Determine if strategy is long premium (pays for options).

        Args:
            strategy_type: Type of strategy

        Returns:
            True if long premium strategy
        """
        if strategy_type:
            long_premium_types = [
                StrategyType.LONG_STRADDLE,
                StrategyType.LONG_STRANGLE,
                StrategyType.LONG_CALL,
                StrategyType.LONG_PUT,
                StrategyType.BUTTERFLY,
                StrategyType.CALENDAR_CALL,
                StrategyType.CALENDAR_PUT
            ]
            return strategy_type in long_premium_types

        return False


def get_strategy_reward_calculator(strategy_type: StrategyType) -> RewardCalculator:
    """
    Factory function to get appropriate reward calculator for strategy type.

    Args:
        strategy_type: Type of options strategy

    Returns:
        Appropriate reward calculator instance
    """
    # Neutral strategies
    neutral_strategies = [
        StrategyType.IRON_CONDOR,
        StrategyType.BUTTERFLY,
        StrategyType.SHORT_STRADDLE,
        StrategyType.SHORT_STRANGLE
    ]

    # Directional strategies
    directional_strategies = [
        StrategyType.LONG_CALL,
        StrategyType.SHORT_CALL,
        StrategyType.LONG_PUT,
        StrategyType.SHORT_PUT,
        StrategyType.BULL_CALL_SPREAD,
        StrategyType.BEAR_CALL_SPREAD,
        StrategyType.BULL_PUT_SPREAD,
        StrategyType.BEAR_PUT_SPREAD
    ]

    # Volatility strategies
    volatility_strategies = [
        StrategyType.LONG_STRADDLE,
        StrategyType.LONG_STRANGLE,
        StrategyType.CALENDAR_CALL,
        StrategyType.CALENDAR_PUT
    ]

    if strategy_type in neutral_strategies:
        return NeutralStrategyReward()
    elif strategy_type in directional_strategies:
        return DirectionalStrategyReward()
    elif strategy_type in volatility_strategies:
        return VolatilityStrategyReward()
    else:
        # Default to neutral calculator
        return NeutralStrategyReward()

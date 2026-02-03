"""
Base environment class for options position management.

Implements OpenAI Gym interface for reinforcement learning with options positions.
"""

import gym
from gym import spaces
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ..strategies.base import StrategyType
from .actions import ActionType, PositionAdjustment, ActionValidator, calculate_action_cost


@dataclass
class PositionState:
    """Current state of an options position."""
    strategy_type: StrategyType
    strikes: np.ndarray  # Strike prices for the position
    quantities: np.ndarray  # Quantities for each leg
    entry_price: float  # Price when position was entered
    current_price: float  # Current underlying price
    days_to_expiry: int  # Days remaining until expiration
    entry_iv: float  # Implied volatility at entry
    current_iv: float  # Current implied volatility
    unrealized_pnl: float  # Current P/L
    max_loss: float  # Maximum potential loss
    max_profit: float  # Maximum potential profit


class OptionsEnvironment(gym.Env):
    """
    OpenAI Gym environment for options position management.

    Observation space: 30-dimensional continuous vector
    - Price movements (10 dims): Recent price history and changes
    - Option Greeks (5 dims): Delta, Gamma, Theta, Vega, Rho
    - Position metrics (5 dims): P/L, days to expiry, position size, etc.
    - Time decay (2 dims): Theta decay rate, time value remaining
    - Volatility (3 dims): Current IV, IV rank, IV change
    - Regime indicators (5 dims): Trend, momentum, volatility regime

    Action space: 4 discrete actions
    - 0: HOLD - Keep position unchanged
    - 1: CLOSE - Close entire position
    - 2: ADJUST - Modify strikes or quantities
    - 3: ROLL - Move to next expiration
    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 initial_capital: float = 100000,
                 max_steps: int = 252,  # Trading days in a year
                 seed: Optional[int] = None):
        """
        Initialize the environment.

        Args:
            initial_capital: Starting capital for trading
            max_steps: Maximum steps in an episode
            seed: Random seed for reproducibility
        """
        super().__init__()

        # Define observation space - 30 dimensional continuous
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(30,),
            dtype=np.float32
        )

        # Define action space - 4 discrete actions
        self.action_space = spaces.Discrete(4)

        # Environment parameters
        self.initial_capital = initial_capital
        self.max_steps = max_steps
        self.current_step = 0

        # Position tracking
        self.position_state: Optional[PositionState] = None
        self.capital = initial_capital

        # Set random seed if provided
        if seed is not None:
            self.seed(seed)

    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.

        Returns:
            observation: Initial observation vector
        """
        # Reset step counter
        self.current_step = 0
        self.capital = self.initial_capital

        # Initialize a random position for training
        self.position_state = self._create_initial_position()

        # Generate initial observation
        observation = self._get_observation()

        return observation

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0-3)

        Returns:
            observation: New observation after action
            reward: Reward from the action
            done: Whether episode is finished
            info: Additional information
        """
        if self.position_state is None:
            raise RuntimeError("Environment must be reset before stepping")

        # Validate action
        action_type = ActionType(action)
        is_valid, invalid_reason = ActionValidator.is_action_valid(
            action_type, self.position_state, self.current_step, self.max_steps
        )

        if not is_valid:
            # Invalid action gets negative reward
            reward = -10.0
            info = {'invalid_action': True, 'reason': invalid_reason}
        else:
            # Execute the action
            reward = self._execute_action(action_type)
            info = {'invalid_action': False}

        # Update market state (simplified simulation)
        self._update_market_state()

        # Increment step counter
        self.current_step += 1

        # Check terminal conditions
        done = self._is_done()

        # Update observation
        observation = self._get_observation()

        # Add additional info
        info.update({
            'position_pnl': self.position_state.unrealized_pnl if self.position_state else 0,
            'capital': self.capital,
            'step': self.current_step,
            'action_type': action_type.name
        })

        return observation, reward, done, info

    def _execute_action(self, action: ActionType) -> float:
        """
        Execute the given action and return immediate reward.

        Args:
            action: Action to execute

        Returns:
            Immediate reward from the action
        """
        pos = self.position_state
        action_cost = calculate_action_cost(action, pos)

        if action == ActionType.HOLD:
            # Small positive reward for holding profitable positions
            if pos.unrealized_pnl > 0:
                reward = pos.unrealized_pnl * 0.01  # 1% of P/L as reward
            else:
                reward = pos.unrealized_pnl * 0.005  # Smaller penalty for losses

        elif action == ActionType.CLOSE:
            # Realize the P/L
            reward = pos.unrealized_pnl + action_cost
            self.capital += pos.unrealized_pnl
            self.position_state = None  # Position is closed

        elif action == ActionType.ADJUST:
            # Adjust the position
            adjustment = PositionAdjustment.adjust_position(pos, pos.current_price)
            pos.strikes = adjustment['new_strikes']
            adjustment_cost = adjustment['cost'] + action_cost
            reward = -adjustment_cost  # Cost of adjustment
            self.capital -= adjustment_cost

        elif action == ActionType.ROLL:
            # Roll to next expiration
            roll = PositionAdjustment.roll_position(pos)
            pos.days_to_expiry = roll['new_expiry']
            roll_cost = roll['cost'] + action_cost
            reward = -roll_cost  # Cost of rolling
            self.capital -= roll_cost
        else:
            reward = 0.0

        return reward

    def _update_market_state(self):
        """Update market prices and Greeks (simplified simulation)."""
        if self.position_state is None:
            return

        pos = self.position_state

        # Simple random walk for price
        if hasattr(self, 'np_random'):
            price_change = self.np_random.normal(0, pos.current_price * 0.01)
        else:
            price_change = np.random.normal(0, pos.current_price * 0.01)

        pos.current_price += price_change

        # Update days to expiry
        pos.days_to_expiry = max(0, pos.days_to_expiry - 1)

        # Simple P/L calculation based on price movement
        price_move_pct = (pos.current_price - pos.entry_price) / pos.entry_price

        # Simplified P/L for Iron Condor (profitable in range)
        if pos.strategy_type == StrategyType.IRON_CONDOR:
            # Profitable if price stays between inner strikes
            lower_strike = pos.strikes[1]
            upper_strike = pos.strikes[2]

            if lower_strike <= pos.current_price <= upper_strike:
                # In profit zone
                pos.unrealized_pnl = pos.max_profit * (1 - pos.days_to_expiry / 30)
            else:
                # Outside profit zone
                distance = min(abs(pos.current_price - lower_strike),
                             abs(pos.current_price - upper_strike))
                pos.unrealized_pnl = -distance * 10  # Simplified loss calculation

        # Update IV (random walk)
        if hasattr(self, 'np_random'):
            iv_change = self.np_random.normal(0, 0.01)
        else:
            iv_change = np.random.normal(0, 0.01)
        pos.current_iv = max(0.05, pos.current_iv + iv_change)

    def _is_done(self) -> bool:
        """
        Check if episode is finished.

        Terminal conditions:
        - Position closed
        - Max steps reached
        - Days to expiry <= 0
        - Max loss exceeded
        """
        # No position means it was closed
        if self.position_state is None:
            return True

        # Max steps reached
        if self.current_step >= self.max_steps:
            return True

        # Option expired
        if self.position_state.days_to_expiry <= 0:
            return True

        # Max loss exceeded (risk management)
        if self.position_state.unrealized_pnl < self.position_state.max_loss * 1.5:
            return True

        return False

    def _create_initial_position(self) -> PositionState:
        """Create a random initial position for training."""
        # Simple Iron Condor as example
        position = PositionState(
            strategy_type=StrategyType.IRON_CONDOR,
            strikes=np.array([95, 100, 110, 115]),  # Example strikes
            quantities=np.array([1, -1, -1, 1]),  # Iron Condor quantities
            entry_price=105.0,
            current_price=105.0,
            days_to_expiry=30,
            entry_iv=0.25,
            current_iv=0.25,
            unrealized_pnl=0.0,
            max_loss=-500.0,
            max_profit=200.0
        )
        return position

    def _get_observation(self) -> np.ndarray:
        """
        Generate observation vector from current state.

        Returns:
            30-dimensional observation vector
        """
        obs = np.zeros(30, dtype=np.float32)

        if self.position_state is None:
            return obs

        pos = self.position_state

        # Price movements (10 dims)
        obs[0] = pos.current_price / pos.entry_price - 1  # Price return
        obs[1] = pos.current_price / 100  # Normalized price
        obs[2:10] = 0.0  # Placeholder for price history

        # Option Greeks (5 dims) - placeholders
        obs[10:15] = 0.0

        # Position metrics (5 dims)
        obs[15] = pos.unrealized_pnl / abs(pos.max_loss) if pos.max_loss != 0 else 0
        obs[16] = pos.days_to_expiry / 30  # Normalized days to expiry
        obs[17] = len(pos.strikes) / 4  # Number of legs normalized
        obs[18] = pos.max_profit / abs(pos.max_loss) if pos.max_loss != 0 else 0
        obs[19] = 0.0  # Placeholder for position delta

        # Time decay (2 dims)
        obs[20] = pos.days_to_expiry / 30
        obs[21] = max(0, 1 - pos.days_to_expiry / 30)  # Time decay factor

        # Volatility (3 dims)
        obs[22] = pos.current_iv
        obs[23] = pos.current_iv / pos.entry_iv - 1 if pos.entry_iv != 0 else 0
        obs[24] = 0.0  # Placeholder for IV rank

        # Regime indicators (5 dims) - placeholders
        obs[25:30] = 0.0

        return obs

    def seed(self, seed: int = None):
        """Set random seed for reproducibility."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def render(self, mode: str = 'human'):
        """Render the environment (text output for now)."""
        if mode == 'human' and self.position_state:
            pos = self.position_state
            print(f"\n=== Position State ===")
            print(f"Strategy: {pos.strategy_type.value}")
            print(f"Price: {pos.current_price:.2f} (entry: {pos.entry_price:.2f})")
            print(f"P/L: {pos.unrealized_pnl:.2f}")
            print(f"Days to expiry: {pos.days_to_expiry}")
            print(f"IV: {pos.current_iv:.2%}")
            print(f"Step: {self.current_step}/{self.max_steps}")

    def close(self):
        """Clean up environment resources."""
        pass
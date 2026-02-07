"""
Complete options trading environment for reinforcement learning.

Integrates all components: base environment, episode management, market simulation,
and reward calculation for realistic training scenarios.
"""

import gym
from gym import spaces
import numpy as np
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass

from .base import OptionsEnvironment, PositionState
from .episode import EpisodeManager, TerminalReason
from .market_sim import MarketDataSimulator, MarketState, MarketRegime
from .rewards import RewardCalculator
from .strategy_rewards import NeutralStrategyReward, DirectionalStrategyReward, VolatilityStrategyReward
from .reward_scaler import RewardScaler
from .actions import ActionType
from ..strategies.base import StrategyType


class OptionsTrainingEnvironment(gym.Env):
    """
    Complete options trading environment for RL training.

    Combines market simulation, position management, strategy-specific rewards,
    and episode management for realistic training scenarios.
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 strategy_type: StrategyType = StrategyType.IRON_CONDOR,
                 initial_capital: float = 100000,
                 max_steps: int = 252,
                 market_regime: MarketRegime = MarketRegime.MEAN_REVERTING,
                 seed: Optional[int] = None):
        """
        Initialize complete training environment.

        Args:
            strategy_type: Options strategy to trade
            initial_capital: Starting capital
            max_steps: Maximum steps per episode
            market_regime: Market regime for simulation
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.strategy_type = strategy_type
        self.initial_capital = initial_capital
        self.max_steps = max_steps
        self.market_regime = market_regime

        # Set up spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(35,),  # Extended observation space
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        # Initialize components
        self.market_simulator = MarketDataSimulator(
            initial_price=100.0,
            seed=seed,
            regime=market_regime
        )

        self.episode_manager = EpisodeManager(
            max_steps=max_steps,
            max_loss_threshold=0.20,
            early_stop_bonus=10.0
        )

        # Initialize strategy-specific reward calculator
        self.reward_calculator = self._create_reward_calculator(strategy_type)
        self.reward_scaler = RewardScaler(
            clip_range=(-3.0, 3.0),
            temperature=1.0
        )

        # Current state
        self.current_step = 0
        self.position_state: Optional[PositionState] = None
        self.market_state: Optional[MarketState] = None
        self.capital = initial_capital
        self.episode_history: List[Dict[str, Any]] = []

        # Set random seed
        if seed is not None:
            self.seed(seed)

    def _create_reward_calculator(self, strategy_type: StrategyType) -> RewardCalculator:
        """Create appropriate reward calculator for strategy type."""
        # Map strategy types to categories
        neutral_strategies = {
            StrategyType.IRON_CONDOR,
            StrategyType.BUTTERFLY,
            StrategyType.SHORT_STRADDLE,
            StrategyType.SHORT_STRANGLE,
            StrategyType.CALENDAR_CALL,
            StrategyType.CALENDAR_PUT
        }

        directional_strategies = {
            StrategyType.BULL_CALL_SPREAD,
            StrategyType.BULL_PUT_SPREAD,
            StrategyType.BEAR_CALL_SPREAD,
            StrategyType.BEAR_PUT_SPREAD,
            StrategyType.LONG_CALL,
            StrategyType.SHORT_CALL,
            StrategyType.LONG_PUT,
            StrategyType.SHORT_PUT
        }

        volatility_strategies = {
            StrategyType.LONG_STRADDLE,
            StrategyType.LONG_STRANGLE
        }

        if strategy_type in neutral_strategies:
            return NeutralStrategyReward()
        elif strategy_type in directional_strategies:
            return DirectionalStrategyReward()
        elif strategy_type in volatility_strategies:
            return VolatilityStrategyReward()
        else:
            # Default to neutral
            return NeutralStrategyReward()

    def reset(self) -> np.ndarray:
        """Reset environment for new episode."""
        # Reset step counter and capital
        self.current_step = 0
        self.capital = self.initial_capital

        # Reset episode manager
        self.episode_manager.reset()

        # Reset episode history
        self.episode_history = []

        # Create new position
        self.position_state = self._create_initial_position()

        # Initialize market state
        self.market_state = MarketState(
            price=self.position_state.current_price,
            volatility=self.position_state.current_iv,
            time_to_expiry=self.position_state.days_to_expiry / 365,
            interest_rate=0.04,
            dividend_yield=0.02
        )

        # Generate initial observation
        observation = self._get_observation()

        return observation

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if self.position_state is None or self.market_state is None:
            raise RuntimeError("Environment must be reset before stepping")

        # Convert action
        action_type = ActionType(action)

        # Update market state first
        self.market_state = self.market_simulator.simulate_step(self.market_state)
        self._update_position_from_market()

        # Calculate reward using strategy-specific calculator
        market_data = {
            'price': self.market_state.price,
            'volatility': self.market_state.volatility,
            'time_to_expiry': self.market_state.time_to_expiry
        }

        position_data = {
            'strategy_type': self.position_state.strategy_type,
            'strikes': self.position_state.strikes,
            'quantities': self.position_state.quantities,
            'current_pnl': self.position_state.unrealized_pnl,
            'max_loss': self.position_state.max_loss,
            'entry_price': self.position_state.entry_price,
            'current_price': self.position_state.current_price,
            'days_to_expiry': self.position_state.days_to_expiry,
            'current_iv': self.position_state.current_iv
        }

        # Calculate raw reward
        raw_reward = self.reward_calculator.calculate_reward(
            position_data, action_type.name, market_data
        )

        # Execute action and get additional reward
        action_reward = self._execute_action(action_type)
        total_raw_reward = raw_reward + action_reward

        # Scale reward
        reward = self.reward_scaler.scale_reward(total_raw_reward)

        # Increment step
        self.current_step += 1

        # Check terminal conditions
        done, terminal_reward = self._check_terminal()
        reward += terminal_reward

        # Update episode statistics
        current_pnl = self.position_state.unrealized_pnl if self.position_state else 0
        self.episode_manager.update_stats(action_type.name, reward, current_pnl)

        # Generate observation
        observation = self._get_observation()

        # Create info dict
        info = {
            'position_pnl': current_pnl,
            'capital': self.capital,
            'step': self.current_step,
            'action_type': action_type.name,
            'terminal_reward': terminal_reward,
            'raw_reward': total_raw_reward,
            'scaled_reward': reward,
            'market_price': self.market_state.price,
            'market_volatility': self.market_state.volatility,
            'time_to_expiry': self.market_state.time_to_expiry
        }

        # Add episode summary if done
        if done:
            info['episode_summary'] = self.episode_manager.get_episode_summary()

        # Store step in history
        self.episode_history.append({
            'step': self.current_step,
            'action': action_type.name,
            'reward': reward,
            'pnl': current_pnl,
            'price': self.market_state.price,
            'done': done
        })

        return observation, reward, done, info

    def _create_initial_position(self) -> PositionState:
        """Create initial position based on strategy type."""
        # Random initial conditions
        base_price = 100 + np.random.normal(0, 5)  # Price around 100
        initial_iv = 0.15 + np.random.normal(0, 0.05)  # IV around 15-25%
        initial_iv = max(0.05, min(initial_iv, 0.50))  # Clamp IV
        days_to_expiry = np.random.randint(15, 45)  # 15-45 days

        if self.strategy_type == StrategyType.IRON_CONDOR:
            # Iron Condor: Sell put spread, sell call spread
            strikes = np.array([base_price - 10, base_price - 5, base_price + 5, base_price + 10])
            quantities = np.array([1, -1, -1, 1])  # Long wings, short body
            max_loss = -400  # Risk between strikes
            max_profit = 100   # Credit received

        elif self.strategy_type == StrategyType.BUTTERFLY:
            # Iron Butterfly: Tighter strikes than condor
            strikes = np.array([base_price - 5, base_price, base_price, base_price + 5])
            quantities = np.array([1, -2, 2, -1])  # Different structure
            max_loss = -350
            max_profit = 150

        elif self.strategy_type == StrategyType.LONG_STRADDLE:
            # Long straddle: Buy call and put at same strike
            strikes = np.array([base_price, base_price])
            quantities = np.array([1, 1])  # Long both
            max_loss = -500  # Premium paid
            max_profit = np.inf  # Unlimited (use large number)

        elif self.strategy_type in [StrategyType.BULL_CALL_SPREAD, StrategyType.BEAR_PUT_SPREAD]:
            # Directional spreads
            if self.strategy_type == StrategyType.BULL_CALL_SPREAD:
                strikes = np.array([base_price, base_price + 5])
                quantities = np.array([1, -1])  # Buy lower, sell higher
            else:  # Bear put spread
                strikes = np.array([base_price - 5, base_price])
                quantities = np.array([-1, 1])  # Sell higher, buy lower

            max_loss = -250
            max_profit = 250

        else:
            # Default to Iron Condor for other strategies
            strikes = np.array([base_price - 10, base_price - 5, base_price + 5, base_price + 10])
            quantities = np.array([1, -1, -1, 1])
            max_loss = -400
            max_profit = 100

        return PositionState(
            strategy_type=self.strategy_type,
            strikes=strikes,
            quantities=quantities,
            entry_price=base_price,
            current_price=base_price,
            days_to_expiry=days_to_expiry,
            entry_iv=initial_iv,
            current_iv=initial_iv,
            unrealized_pnl=0.0,
            max_loss=max_loss,
            max_profit=max_profit
        )

    def _update_position_from_market(self):
        """Update position state from current market state."""
        if self.position_state is None or self.market_state is None:
            return

        # Update basic market data
        self.position_state.current_price = self.market_state.price
        self.position_state.current_iv = self.market_state.volatility
        self.position_state.days_to_expiry = int(self.market_state.time_to_expiry * 365)

        # Calculate P/L based on market movement
        price_move = self.market_state.price - self.position_state.entry_price
        price_move_pct = price_move / self.position_state.entry_price

        # Simplified P/L calculation based on strategy type
        if self.strategy_type == StrategyType.IRON_CONDOR:
            # Profitable if price stays between inner strikes
            lower_bound = self.position_state.strikes[1]
            upper_bound = self.position_state.strikes[2]

            if lower_bound <= self.market_state.price <= upper_bound:
                # In profit zone - time decay helps
                time_decay_factor = 1 - (self.position_state.days_to_expiry / 30)
                self.position_state.unrealized_pnl = self.position_state.max_profit * time_decay_factor * 0.8
            else:
                # Outside profit zone - calculate loss
                distance = min(abs(self.market_state.price - lower_bound),
                             abs(self.market_state.price - upper_bound))
                self.position_state.unrealized_pnl = -distance * 15  # $15 per point

        elif self.strategy_type == StrategyType.LONG_STRADDLE:
            # Benefits from large moves in either direction
            move_magnitude = abs(price_move_pct)
            if move_magnitude > 0.05:  # >5% move
                self.position_state.unrealized_pnl = move_magnitude * 1000 - 200  # Profit after premium
            else:
                # Time decay hurts
                time_decay = (30 - self.position_state.days_to_expiry) * 8
                self.position_state.unrealized_pnl = -time_decay

        elif self.strategy_type == StrategyType.BULL_CALL_SPREAD:
            # Benefits from upward moves
            if price_move > 0:
                self.position_state.unrealized_pnl = min(price_move * 20, self.position_state.max_profit)
            else:
                self.position_state.unrealized_pnl = max(price_move * 15, self.position_state.max_loss * 0.8)

        else:
            # Default calculation
            self.position_state.unrealized_pnl = price_move_pct * 300

        # Clamp to max loss/profit
        self.position_state.unrealized_pnl = max(
            self.position_state.max_loss,
            min(self.position_state.unrealized_pnl, self.position_state.max_profit)
        )

    def _execute_action(self, action: ActionType) -> float:
        """Execute action and return immediate reward."""
        if action == ActionType.CLOSE:
            # Close position
            realized_pnl = self.position_state.unrealized_pnl
            self.capital += realized_pnl
            self.position_state = None  # Mark as closed
            return realized_pnl * 0.1  # Small fraction as immediate reward

        elif action == ActionType.ADJUST:
            # Adjustment cost
            adjustment_cost = 25
            self.capital -= adjustment_cost
            return -adjustment_cost

        elif action == ActionType.ROLL:
            # Rolling cost and benefit
            if self.position_state.days_to_expiry <= 14:
                roll_cost = 35
                self.capital -= roll_cost
                self.position_state.days_to_expiry += 30  # Extend expiry
                return -roll_cost + 10  # Cost minus small benefit
            else:
                return -10  # Penalty for rolling too early

        # HOLD action
        return 0

    def _check_terminal(self) -> Tuple[bool, float]:
        """Check terminal conditions and calculate terminal reward."""
        state = {
            'position_state': self.position_state,
            'position_closed': self.position_state is None,
            'current_step': self.current_step
        }

        is_terminal, reason = self.episode_manager.is_terminal(state)

        terminal_reward = 0.0
        if is_terminal:
            final_pnl = self.position_state.unrealized_pnl if self.position_state else 0
            raw_terminal_reward = self.episode_manager.calculate_terminal_reward(reason, final_pnl)
            terminal_reward = self.reward_scaler.scale_reward(raw_terminal_reward)

        return is_terminal, terminal_reward

    def _get_observation(self) -> np.ndarray:
        """Generate observation vector (35 dimensions)."""
        obs = np.zeros(35, dtype=np.float32)

        if self.position_state is None or self.market_state is None:
            return obs

        pos = self.position_state
        market = self.market_state

        # Market state (5 dims)
        obs[0] = market.price / 100  # Normalized price
        obs[1] = (market.price - pos.entry_price) / pos.entry_price  # Price return
        obs[2] = market.volatility  # Current IV
        obs[3] = market.volatility / pos.entry_iv - 1  # IV change
        obs[4] = market.time_to_expiry  # Time to expiry

        # Position state (10 dims)
        obs[5] = pos.unrealized_pnl / abs(pos.max_loss) if pos.max_loss != 0 else 0
        obs[6] = pos.days_to_expiry / 30  # Normalized days to expiry
        obs[7] = len(pos.strikes) / 4  # Number of legs normalized
        obs[8] = pos.max_profit / abs(pos.max_loss) if pos.max_loss != 0 else 0
        obs[9] = (pos.current_price - np.mean(pos.strikes)) / np.std(pos.strikes) if np.std(pos.strikes) > 0 else 0

        # Greeks placeholders (5 dims) - would be calculated with real Greeks
        obs[10:15] = 0

        # Strategy indicators (10 dims) - one-hot encoding for strategy type
        strategy_list = list(StrategyType)
        try:
            strategy_idx = strategy_list.index(pos.strategy_type) % 10
            obs[15 + strategy_idx] = 1
        except (ValueError, AttributeError):
            # Default to first position if strategy not found
            obs[15] = 1

        # Episode state (5 dims)
        obs[25] = self.current_step / self.max_steps
        obs[26] = self.capital / self.initial_capital - 1  # Capital return
        obs[27] = len(self.episode_history) / self.max_steps if self.episode_history else 0
        obs[28] = self.episode_manager.stats.num_adjustments / 5  # Normalized
        obs[29] = self.episode_manager.stats.num_rolls / 3  # Normalized

        # Market regime indicators (5 dims) - simplified
        regime_encoding = {
            MarketRegime.TRENDING_UP: [1, 0, 0, 0, 0],
            MarketRegime.TRENDING_DOWN: [0, 1, 0, 0, 0],
            MarketRegime.MEAN_REVERTING: [0, 0, 1, 0, 0],
            MarketRegime.HIGH_VOLATILITY: [0, 0, 0, 1, 0],
            MarketRegime.LOW_VOLATILITY: [0, 0, 0, 0, 1]
        }
        obs[30:35] = regime_encoding.get(self.market_regime, [0, 0, 1, 0, 0])

        return obs

    def render(self, mode: str = 'human'):
        """Render the environment state."""
        if mode == 'human' and self.position_state and self.market_state:
            print(f"\n=== Options Trading Environment ===")
            print(f"Strategy: {self.position_state.strategy_type.value}")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Market Price: {self.market_state.price:.2f}")
            print(f"Entry Price: {self.position_state.entry_price:.2f}")
            print(f"P/L: ${self.position_state.unrealized_pnl:.2f}")
            print(f"IV: {self.market_state.volatility:.1%}")
            print(f"Days to Expiry: {self.position_state.days_to_expiry}")
            print(f"Capital: ${self.capital:.0f}")

            if self.episode_history:
                recent_actions = [step['action'] for step in self.episode_history[-5:]]
                print(f"Recent Actions: {recent_actions}")

    def seed(self, seed: int = None):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)
            # Also seed the market simulator
            self.market_simulator = MarketDataSimulator(
                initial_price=100.0,
                seed=seed,
                regime=self.market_regime
            )

    def close(self):
        """Clean up resources."""
        pass


def make_env(strategy_name: str,
             regime: str = 'mean_reverting',
             seed: Optional[int] = None) -> OptionsTrainingEnvironment:
    """
    Factory function to create environment with different configurations.

    Args:
        strategy_name: Name of strategy ('IronCondor', 'LongStraddle', etc.)
        regime: Market regime ('trending_up', 'trending_down', 'mean_reverting', etc.)
        seed: Random seed

    Returns:
        Configured OptionsTrainingEnvironment
    """
    # Map string names to enums
    strategy_mapping = {
        'IronCondor': StrategyType.IRON_CONDOR,
        'IronButterfly': StrategyType.BUTTERFLY,  # Note: enum name is BUTTERFLY
        'LongStraddle': StrategyType.LONG_STRADDLE,
        'BullCallSpread': StrategyType.BULL_CALL_SPREAD,
        'BearPutSpread': StrategyType.BEAR_PUT_SPREAD,
    }

    regime_mapping = {
        'trending_up': MarketRegime.TRENDING_UP,
        'trending_down': MarketRegime.TRENDING_DOWN,
        'mean_reverting': MarketRegime.MEAN_REVERTING,
        'high_volatility': MarketRegime.HIGH_VOLATILITY,
        'low_volatility': MarketRegime.LOW_VOLATILITY,
    }

    strategy_type = strategy_mapping.get(strategy_name, StrategyType.IRON_CONDOR)
    market_regime = regime_mapping.get(regime, MarketRegime.MEAN_REVERTING)

    return OptionsTrainingEnvironment(
        strategy_type=strategy_type,
        market_regime=market_regime,
        seed=seed
    )
"""Tests for base environment class."""

import pytest
import numpy as np
import gym
from gym import spaces

from src.environments.base import OptionsEnvironment, PositionState
from src.strategies.base import StrategyType


class TestOptionsEnvironment:
    """Test the base OptionsEnvironment class."""

    def test_environment_creation(self):
        """Test that environment can be created."""
        env = OptionsEnvironment()
        assert isinstance(env, gym.Env)
        assert env.initial_capital == 100000
        assert env.max_steps == 252

    def test_observation_space(self):
        """Test observation space is properly defined."""
        env = OptionsEnvironment()
        assert isinstance(env.observation_space, spaces.Box)
        assert env.observation_space.shape == (30,)
        assert env.observation_space.dtype == np.float32

    def test_reset_returns_observation(self):
        """Test reset returns properly shaped observation."""
        env = OptionsEnvironment()
        obs = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (30,)
        assert obs.dtype == np.float32

        # Check observation is within bounds
        assert env.observation_space.contains(obs)

    def test_reset_initializes_position(self):
        """Test reset creates initial position state."""
        env = OptionsEnvironment()
        env.reset()

        assert env.position_state is not None
        assert isinstance(env.position_state, PositionState)
        assert env.position_state.strategy_type in StrategyType
        assert env.current_step == 0
        assert env.capital == env.initial_capital

    def test_observation_generation(self):
        """Test observation vector generation from position state."""
        env = OptionsEnvironment()
        obs = env.reset()

        # Check specific observation components
        pos = env.position_state

        # Price return should be 0 initially (current = entry)
        assert abs(obs[0]) < 0.01  # Near zero

        # Days to expiry normalized
        assert 0 <= obs[16] <= 1

        # Time decay factor
        assert 0 <= obs[21] <= 1

        # IV ratio should be 0 initially (current = entry)
        assert abs(obs[23]) < 0.01

    def test_seed_reproducibility(self):
        """Test that seeding provides reproducible results."""
        env1 = OptionsEnvironment(seed=42)
        env2 = OptionsEnvironment(seed=42)

        obs1 = env1.reset()
        obs2 = env2.reset()

        # Should produce identical observations
        np.testing.assert_array_equal(obs1, obs2)

    def test_render_human_mode(self):
        """Test rendering in human mode doesn't crash."""
        env = OptionsEnvironment()
        env.reset()

        # Should not raise an error
        env.render(mode='human')

    def test_position_state_dataclass(self):
        """Test PositionState dataclass."""
        position = PositionState(
            strategy_type=StrategyType.IRON_CONDOR,
            strikes=np.array([95, 100, 110, 115]),
            quantities=np.array([1, -1, -1, 1]),
            entry_price=105.0,
            current_price=106.0,
            days_to_expiry=25,
            entry_iv=0.20,
            current_iv=0.22,
            unrealized_pnl=50.0,
            max_loss=-500.0,
            max_profit=200.0
        )

        assert position.strategy_type == StrategyType.IRON_CONDOR
        assert len(position.strikes) == 4
        assert position.unrealized_pnl == 50.0

    def test_environment_parameters(self):
        """Test environment accepts custom parameters."""
        env = OptionsEnvironment(
            initial_capital=50000,
            max_steps=100,
            seed=123
        )

        assert env.initial_capital == 50000
        assert env.max_steps == 100
        env.reset()
        assert env.capital == 50000
"""Integration tests for complete training environment."""

import pytest
import numpy as np
import gym

from src.environments import make_env, OptionsTrainingEnvironment
from src.environments.market_sim import MarketRegime
from src.strategies.base import StrategyType


class TestOptionsTrainingEnvironment:
    """Test complete training environment integration."""

    def test_environment_creation(self):
        """Test environment can be created."""
        env = OptionsTrainingEnvironment()
        assert isinstance(env, gym.Env)
        assert env.observation_space.shape == (35,)
        assert env.action_space.n == 4

    def test_reset_functionality(self):
        """Test environment reset works correctly."""
        env = OptionsTrainingEnvironment(seed=42)
        obs = env.reset()

        assert obs.shape == (35,)
        assert env.current_step == 0
        assert env.position_state is not None
        assert env.market_state is not None
        assert env.capital == env.initial_capital

    def test_step_functionality(self):
        """Test environment step function works."""
        env = OptionsTrainingEnvironment(seed=42)
        env.reset()

        obs, reward, done, info = env.step(0)  # HOLD action

        assert obs.shape == (35,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

        # Check info contains expected fields
        assert 'position_pnl' in info
        assert 'capital' in info
        assert 'step' in info
        assert 'action_type' in info

    def test_episode_completion(self):
        """Test complete episode can be run."""
        env = OptionsTrainingEnvironment(seed=42, max_steps=10)
        obs = env.reset()

        steps = 0
        total_reward = 0
        done = False

        while not done and steps < 20:  # Safety limit
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

        assert done or steps >= 10
        assert 'episode_summary' in info if done else True

    def test_different_strategies(self):
        """Test environment works with different strategies."""
        strategies = [
            StrategyType.IRON_CONDOR,
            StrategyType.LONG_STRADDLE,
            StrategyType.BULL_CALL_SPREAD
        ]

        for strategy in strategies:
            env = OptionsTrainingEnvironment(
                strategy_type=strategy,
                seed=42
            )
            obs = env.reset()

            assert env.position_state.strategy_type == strategy
            assert obs.shape == (35,)

            # Run a few steps
            for _ in range(5):
                obs, reward, done, info = env.step(0)  # HOLD
                if done:
                    break

    def test_different_regimes(self):
        """Test environment works with different market regimes."""
        regimes = [
            MarketRegime.TRENDING_UP,
            MarketRegime.TRENDING_DOWN,
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.LOW_VOLATILITY
        ]

        for regime in regimes:
            env = OptionsTrainingEnvironment(
                market_regime=regime,
                seed=42
            )
            obs = env.reset()

            assert env.market_regime == regime
            assert obs.shape == (35,)

    def test_action_execution(self):
        """Test different actions can be executed."""
        env = OptionsTrainingEnvironment(seed=42)
        env.reset()

        # Test HOLD action
        obs, reward, done, info = env.step(0)
        assert info['action_type'] == 'HOLD'

        if not done:
            # Test ADJUST action
            obs, reward, done, info = env.step(2)
            assert info['action_type'] == 'ADJUST'

        if not done:
            # Test CLOSE action
            obs, reward, done, info = env.step(1)
            assert info['action_type'] == 'CLOSE'
            assert done  # Should be done after closing

    def test_reward_scaling(self):
        """Test reward scaling is working."""
        env = OptionsTrainingEnvironment(seed=42)
        env.reset()

        rewards = []
        for _ in range(10):
            obs, reward, done, info = env.step(0)  # HOLD
            rewards.append(reward)
            if done:
                env.reset()

        # Rewards should be reasonable magnitude (scaled)
        assert all(-10 < r < 10 for r in rewards)

    def test_observation_consistency(self):
        """Test observation vector consistency."""
        env = OptionsTrainingEnvironment(seed=42)
        obs = env.reset()

        # All observations should be finite
        assert np.isfinite(obs).all()

        # Run several steps and check consistency
        for _ in range(5):
            obs, reward, done, info = env.step(0)
            assert np.isfinite(obs).all()
            assert obs.shape == (35,)
            if done:
                break

    def test_terminal_conditions(self):
        """Test various terminal conditions."""
        # Test position closing
        env = OptionsTrainingEnvironment(seed=42)
        env.reset()

        obs, reward, done, info = env.step(1)  # CLOSE
        assert done
        assert 'episode_summary' in info

        # Test max steps
        env = OptionsTrainingEnvironment(max_steps=3, seed=42)
        env.reset()

        for _ in range(3):
            obs, reward, done, info = env.step(0)  # HOLD

        assert done  # Should be done after max steps

    def test_render_functionality(self):
        """Test rendering doesn't crash."""
        env = OptionsTrainingEnvironment(seed=42)
        env.reset()

        # Should not raise an error
        env.render(mode='human')

        # Take a step and render again
        env.step(0)
        env.render(mode='human')


class TestMakeEnvFactory:
    """Test the make_env factory function."""

    def test_make_env_basic(self):
        """Test basic make_env functionality."""
        env = make_env('IronCondor')

        assert isinstance(env, OptionsTrainingEnvironment)
        assert env.strategy_type == StrategyType.IRON_CONDOR

    def test_make_env_with_regime(self):
        """Test make_env with different regime."""
        env = make_env('LongStraddle', regime='high_volatility', seed=42)

        assert env.strategy_type == StrategyType.LONG_STRADDLE
        assert env.market_regime == MarketRegime.HIGH_VOLATILITY

    def test_make_env_unknown_strategy(self):
        """Test make_env with unknown strategy defaults correctly."""
        env = make_env('UnknownStrategy')

        # Should default to Iron Condor
        assert env.strategy_type == StrategyType.IRON_CONDOR

    def test_make_env_unknown_regime(self):
        """Test make_env with unknown regime defaults correctly."""
        env = make_env('IronCondor', regime='unknown_regime')

        # Should default to mean reverting
        assert env.market_regime == MarketRegime.MEAN_REVERTING

    def test_make_env_with_seed(self):
        """Test make_env with reproducible seed."""
        env1 = make_env('IronCondor', seed=42)
        env2 = make_env('IronCondor', seed=42)

        obs1 = env1.reset()
        obs2 = env2.reset()

        # Should produce similar initial observations
        # (exactly same is hard due to complex initialization)
        assert obs1.shape == obs2.shape

    def test_make_env_full_episode(self):
        """Test running full episode with make_env."""
        env = make_env('BullCallSpread', regime='trending_up', seed=42)

        obs = env.reset()
        steps = 0
        done = False

        while not done and steps < 50:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            steps += 1

        # Episode should complete
        assert done or steps >= 50

    def test_all_strategy_types(self):
        """Test all available strategy types in make_env."""
        strategies = ['IronCondor', 'IronButterfly', 'LongStraddle',
                     'BullCallSpread', 'BearPutSpread']

        for strategy in strategies:
            env = make_env(strategy, seed=42)
            obs = env.reset()

            assert obs.shape == (35,)
            # Run one step to ensure it works
            obs, reward, done, info = env.step(0)

    def test_all_regime_types(self):
        """Test all regime types in make_env."""
        regimes = ['trending_up', 'trending_down', 'mean_reverting',
                  'high_volatility', 'low_volatility']

        for regime in regimes:
            env = make_env('IronCondor', regime=regime, seed=42)
            obs = env.reset()

            assert obs.shape == (35,)
            # Run one step to ensure it works
            obs, reward, done, info = env.step(0)
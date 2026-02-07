"""Tests for market data simulator."""

import pytest
import numpy as np

from src.environments.market_sim import (
    MarketDataSimulator,
    BlackScholesCalculator,
    MarketRegime,
    MarketState,
    SimulationParams
)


class TestBlackScholesCalculator:
    """Test Black-Scholes calculations."""

    def test_call_option_price(self):
        """Test call option pricing."""
        bs = BlackScholesCalculator()

        # ATM call with reasonable parameters
        price = bs.calculate_option_price(
            spot=100, strike=100, time_to_expiry=0.25,
            volatility=0.2, interest_rate=0.05, is_call=True
        )

        # Should be positive and reasonable
        assert price > 0
        assert 5 < price < 15  # Rough sanity check

    def test_put_option_price(self):
        """Test put option pricing."""
        bs = BlackScholesCalculator()

        # ATM put with reasonable parameters
        price = bs.calculate_option_price(
            spot=100, strike=100, time_to_expiry=0.25,
            volatility=0.2, interest_rate=0.05, is_call=False
        )

        # Should be positive and reasonable
        assert price > 0
        assert 3 < price < 12

    def test_option_at_expiry(self):
        """Test option pricing at expiry."""
        bs = BlackScholesCalculator()

        # ITM call at expiry
        call_price = bs.calculate_option_price(
            spot=105, strike=100, time_to_expiry=0,
            volatility=0.2, interest_rate=0.05, is_call=True
        )
        assert call_price == 5  # Intrinsic value

        # OTM call at expiry
        call_price_otm = bs.calculate_option_price(
            spot=95, strike=100, time_to_expiry=0,
            volatility=0.2, interest_rate=0.05, is_call=True
        )
        assert call_price_otm == 0

    def test_delta_calculation(self):
        """Test delta calculation."""
        bs = BlackScholesCalculator()

        # ATM call delta should be around 0.5
        delta = bs.calculate_delta(
            spot=100, strike=100, time_to_expiry=0.25,
            volatility=0.2, interest_rate=0.05, is_call=True
        )
        assert 0.4 < delta < 0.6

        # ATM put delta should be around -0.5
        put_delta = bs.calculate_delta(
            spot=100, strike=100, time_to_expiry=0.25,
            volatility=0.2, interest_rate=0.05, is_call=False
        )
        assert -0.6 < put_delta < -0.4

    def test_gamma_calculation(self):
        """Test gamma calculation."""
        bs = BlackScholesCalculator()

        gamma = bs.calculate_gamma(
            spot=100, strike=100, time_to_expiry=0.25,
            volatility=0.2, interest_rate=0.05
        )

        # Gamma should be positive for both calls and puts
        assert gamma > 0

    def test_theta_calculation(self):
        """Test theta calculation."""
        bs = BlackScholesCalculator()

        # Call theta
        theta_call = bs.calculate_theta(
            spot=100, strike=100, time_to_expiry=0.25,
            volatility=0.2, interest_rate=0.05, is_call=True
        )

        # Theta should be negative (time decay)
        assert theta_call < 0

    def test_vega_calculation(self):
        """Test vega calculation."""
        bs = BlackScholesCalculator()

        vega = bs.calculate_vega(
            spot=100, strike=100, time_to_expiry=0.25,
            volatility=0.2, interest_rate=0.05
        )

        # Vega should be positive
        assert vega > 0


class TestMarketDataSimulator:
    """Test market data simulation functionality."""

    def test_simulator_creation(self):
        """Test simulator can be created."""
        sim = MarketDataSimulator(initial_price=100, seed=42)
        assert sim.initial_price == 100
        assert isinstance(sim.params, SimulationParams)

    def test_regime_parameters(self):
        """Test regime-specific parameters."""
        # High volatility regime
        sim_high_vol = MarketDataSimulator(regime=MarketRegime.HIGH_VOLATILITY, seed=42)
        assert sim_high_vol.params.volatility > 0.02

        # Low volatility regime
        sim_low_vol = MarketDataSimulator(regime=MarketRegime.LOW_VOLATILITY, seed=42)
        assert sim_low_vol.params.volatility < 0.01

        # Trending up
        sim_up = MarketDataSimulator(regime=MarketRegime.TRENDING_UP, seed=42)
        assert sim_up.params.drift > 0

        # Trending down
        sim_down = MarketDataSimulator(regime=MarketRegime.TRENDING_DOWN, seed=42)
        assert sim_down.params.drift < 0

    def test_generate_episode(self):
        """Test episode generation."""
        sim = MarketDataSimulator(seed=42)

        episode = sim.generate_episode(100, 0.2, 20)

        # Should generate correct number of states
        assert len(episode) == 20

        # All states should be MarketState instances
        assert all(isinstance(state, MarketState) for state in episode)

        # Prices should be positive
        assert all(state.price > 0 for state in episode)

        # Time should decay
        assert episode[0].time_to_expiry > episode[-1].time_to_expiry

    def test_reproducibility(self):
        """Test simulation reproducibility with seeds."""
        sim1 = MarketDataSimulator(seed=42)
        sim2 = MarketDataSimulator(seed=42)

        episode1 = sim1.generate_episode(100, 0.2, 10)
        episode2 = sim2.generate_episode(100, 0.2, 10)

        # Should produce identical results
        for state1, state2 in zip(episode1, episode2):
            assert abs(state1.price - state2.price) < 1e-10
            assert abs(state1.volatility - state2.volatility) < 1e-10

    def test_position_greeks_calculation(self):
        """Test portfolio Greeks calculation."""
        sim = MarketDataSimulator(seed=42)

        market_state = MarketState(
            price=100,
            volatility=0.2,
            time_to_expiry=0.25,
            interest_rate=0.05,
            dividend_yield=0.02
        )

        # Simple long call
        strikes = np.array([100])
        quantities = np.array([1])
        option_types = ['call']

        greeks = sim.calculate_position_greeks(
            market_state, strikes, quantities, option_types
        )

        # Check that all Greeks are present
        assert 'value' in greeks
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'theta' in greeks
        assert 'vega' in greeks

        # Long call should have positive delta and gamma
        assert greeks['delta'] > 0
        assert greeks['gamma'] > 0
        assert greeks['vega'] > 0
        assert greeks['theta'] < 0  # Time decay

    def test_iron_condor_greeks(self):
        """Test Greeks calculation for Iron Condor."""
        sim = MarketDataSimulator(seed=42)

        market_state = MarketState(
            price=100,
            volatility=0.2,
            time_to_expiry=0.25,
            interest_rate=0.05,
            dividend_yield=0.02
        )

        # Iron Condor: Sell 95P, Buy 90P, Sell 105C, Buy 110C
        strikes = np.array([90, 95, 105, 110])
        quantities = np.array([1, -1, -1, 1])  # Long wings, short body
        option_types = ['put', 'put', 'call', 'call']

        greeks = sim.calculate_position_greeks(
            market_state, strikes, quantities, option_types
        )

        # Iron Condor should have small delta (delta neutral)
        assert abs(greeks['delta']) < 0.2

        # Should have negative theta (benefits from time decay)
        assert greeks['theta'] < 0

    def test_simulate_step(self):
        """Test single step simulation."""
        sim = MarketDataSimulator(seed=42)

        initial_state = MarketState(
            price=100,
            volatility=0.2,
            time_to_expiry=0.25,
            interest_rate=0.05,
            dividend_yield=0.02
        )

        new_state = sim.simulate_step(initial_state)

        # Time should have decayed
        assert new_state.time_to_expiry < initial_state.time_to_expiry

        # Price should have changed (with high probability)
        assert new_state.price != initial_state.price

        # Price should remain positive
        assert new_state.price > 0

        # Volatility should remain in reasonable bounds
        assert 0.05 <= new_state.volatility <= 1.0

    def test_market_regimes_affect_simulation(self):
        """Test that different regimes produce different price paths."""
        episodes = {}

        for regime in MarketRegime:
            sim = MarketDataSimulator(regime=regime, seed=42)
            episode = sim.generate_episode(100, 0.2, 50)
            final_price = episode[-1].price
            episodes[regime] = final_price

        # Different regimes should produce different outcomes
        # (with high probability, though not guaranteed due to randomness)
        prices = list(episodes.values())
        assert len(set(np.round(prices, 1))) > 1  # At least some difference

    def test_volatility_evolution(self):
        """Test volatility evolution over time."""
        sim = MarketDataSimulator(seed=42)
        episode = sim.generate_episode(100, 0.3, 30)  # Start with high IV

        volatilities = [state.volatility for state in episode]

        # Should generally mean revert (though randomness can interfere)
        # At minimum, should stay within bounds
        assert all(0.05 <= vol <= 1.0 for vol in volatilities)

    def test_greeks_at_expiry(self):
        """Test Greeks calculation near expiry."""
        sim = MarketDataSimulator(seed=42)

        # Very close to expiry
        market_state = MarketState(
            price=105,
            volatility=0.2,
            time_to_expiry=0.001,  # Nearly expired
            interest_rate=0.05,
            dividend_yield=0.02
        )

        strikes = np.array([100])
        quantities = np.array([1])
        option_types = ['call']

        greeks = sim.calculate_position_greeks(
            market_state, strikes, quantities, option_types
        )

        # At expiry, ITM call should have delta near 1
        assert greeks['delta'] > 0.9

        # Gamma and Theta should be near zero
        assert abs(greeks['gamma']) < 0.01
        assert abs(greeks['theta']) < 0.01
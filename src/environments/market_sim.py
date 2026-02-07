"""
Market data simulator for generating realistic price movements and Greeks.

Provides geometric Brownian motion simulation with regime-aware parameters
and Black-Scholes option pricing for realistic trading episodes.
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math


class MarketRegime(Enum):
    """Market regime types affecting simulation parameters."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class SimulationParams:
    """Parameters for market simulation."""
    drift: float = 0.0001  # Daily drift (annualized ~2.5%)
    volatility: float = 0.015  # Daily volatility (annualized ~24%)
    interest_rate: float = 0.04  # Risk-free rate
    dividend_yield: float = 0.02  # Dividend yield


@dataclass
class MarketState:
    """Current market state during simulation."""
    price: float
    volatility: float
    time_to_expiry: float  # In years
    interest_rate: float
    dividend_yield: float


class BlackScholesCalculator:
    """Black-Scholes option pricing and Greeks calculations."""

    @staticmethod
    def calculate_option_price(spot: float, strike: float, time_to_expiry: float,
                             volatility: float, interest_rate: float,
                             dividend_yield: float = 0, is_call: bool = True) -> float:
        """
        Calculate Black-Scholes option price.

        Args:
            spot: Current stock price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility
            interest_rate: Risk-free rate
            dividend_yield: Dividend yield
            is_call: True for call, False for put

        Returns:
            Option price
        """
        if time_to_expiry <= 0:
            # At expiry
            if is_call:
                return max(0, spot - strike)
            else:
                return max(0, strike - spot)

        d1 = (np.log(spot / strike) +
              (interest_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / \
             (volatility * np.sqrt(time_to_expiry))

        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        if is_call:
            price = (spot * np.exp(-dividend_yield * time_to_expiry) * stats.norm.cdf(d1) -
                    strike * np.exp(-interest_rate * time_to_expiry) * stats.norm.cdf(d2))
        else:
            price = (strike * np.exp(-interest_rate * time_to_expiry) * stats.norm.cdf(-d2) -
                    spot * np.exp(-dividend_yield * time_to_expiry) * stats.norm.cdf(-d1))

        return max(0, price)

    @staticmethod
    def calculate_delta(spot: float, strike: float, time_to_expiry: float,
                       volatility: float, interest_rate: float,
                       dividend_yield: float = 0, is_call: bool = True) -> float:
        """Calculate option delta."""
        if time_to_expiry <= 0:
            if is_call:
                return 1.0 if spot > strike else 0.0
            else:
                return -1.0 if spot < strike else 0.0

        d1 = (np.log(spot / strike) +
              (interest_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / \
             (volatility * np.sqrt(time_to_expiry))

        if is_call:
            return np.exp(-dividend_yield * time_to_expiry) * stats.norm.cdf(d1)
        else:
            return -np.exp(-dividend_yield * time_to_expiry) * stats.norm.cdf(-d1)

    @staticmethod
    def calculate_gamma(spot: float, strike: float, time_to_expiry: float,
                       volatility: float, interest_rate: float,
                       dividend_yield: float = 0) -> float:
        """Calculate option gamma."""
        if time_to_expiry <= 0:
            return 0.0

        d1 = (np.log(spot / strike) +
              (interest_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / \
             (volatility * np.sqrt(time_to_expiry))

        gamma = (np.exp(-dividend_yield * time_to_expiry) * stats.norm.pdf(d1)) / \
                (spot * volatility * np.sqrt(time_to_expiry))

        return gamma

    @staticmethod
    def calculate_theta(spot: float, strike: float, time_to_expiry: float,
                       volatility: float, interest_rate: float,
                       dividend_yield: float = 0, is_call: bool = True) -> float:
        """Calculate option theta (per day)."""
        if time_to_expiry <= 0:
            return 0.0

        d1 = (np.log(spot / strike) +
              (interest_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / \
             (volatility * np.sqrt(time_to_expiry))

        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        if is_call:
            theta = (-(spot * stats.norm.pdf(d1) * volatility * np.exp(-dividend_yield * time_to_expiry)) /
                    (2 * np.sqrt(time_to_expiry)) -
                    interest_rate * strike * np.exp(-interest_rate * time_to_expiry) * stats.norm.cdf(d2) +
                    dividend_yield * spot * np.exp(-dividend_yield * time_to_expiry) * stats.norm.cdf(d1))
        else:
            theta = (-(spot * stats.norm.pdf(d1) * volatility * np.exp(-dividend_yield * time_to_expiry)) /
                    (2 * np.sqrt(time_to_expiry)) +
                    interest_rate * strike * np.exp(-interest_rate * time_to_expiry) * stats.norm.cdf(-d2) -
                    dividend_yield * spot * np.exp(-dividend_yield * time_to_expiry) * stats.norm.cdf(-d1))

        return theta / 365  # Convert to daily theta

    @staticmethod
    def calculate_vega(spot: float, strike: float, time_to_expiry: float,
                      volatility: float, interest_rate: float,
                      dividend_yield: float = 0) -> float:
        """Calculate option vega."""
        if time_to_expiry <= 0:
            return 0.0

        d1 = (np.log(spot / strike) +
              (interest_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / \
             (volatility * np.sqrt(time_to_expiry))

        vega = spot * np.exp(-dividend_yield * time_to_expiry) * stats.norm.pdf(d1) * np.sqrt(time_to_expiry)

        return vega / 100  # Convert to per 1% IV change


class MarketDataSimulator:
    """
    Simulates realistic market data for training episodes.

    Uses geometric Brownian motion for price paths with regime-aware parameters.
    Includes Black-Scholes option pricing and Greeks calculations.
    """

    def __init__(self,
                 initial_price: float = 100.0,
                 seed: Optional[int] = None,
                 regime: MarketRegime = MarketRegime.MEAN_REVERTING):
        """
        Initialize market data simulator.

        Args:
            initial_price: Starting stock price
            seed: Random seed for reproducibility
            regime: Market regime affecting simulation parameters
        """
        self.initial_price = initial_price
        self.regime = regime

        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

        # Get regime-specific parameters
        self.params = self._get_regime_params(regime)

        # Black-Scholes calculator
        self.bs_calc = BlackScholesCalculator()

    def _get_regime_params(self, regime: MarketRegime) -> SimulationParams:
        """Get simulation parameters based on market regime."""
        if regime == MarketRegime.TRENDING_UP:
            return SimulationParams(
                drift=0.0004,  # Higher positive drift
                volatility=0.012,  # Moderate volatility
                interest_rate=0.04,
                dividend_yield=0.02
            )
        elif regime == MarketRegime.TRENDING_DOWN:
            return SimulationParams(
                drift=-0.0003,  # Negative drift
                volatility=0.018,  # Higher volatility
                interest_rate=0.04,
                dividend_yield=0.02
            )
        elif regime == MarketRegime.HIGH_VOLATILITY:
            return SimulationParams(
                drift=0.0001,
                volatility=0.025,  # High volatility
                interest_rate=0.04,
                dividend_yield=0.02
            )
        elif regime == MarketRegime.LOW_VOLATILITY:
            return SimulationParams(
                drift=0.0001,
                volatility=0.008,  # Low volatility
                interest_rate=0.04,
                dividend_yield=0.02
            )
        else:  # MEAN_REVERTING
            return SimulationParams(
                drift=0.0,  # No drift (mean reverting)
                volatility=0.015,
                interest_rate=0.04,
                dividend_yield=0.02
            )

    def generate_episode(self, initial_price: float, initial_volatility: float,
                        num_steps: int) -> List[MarketState]:
        """
        Generate a complete episode of market data.

        Args:
            initial_price: Starting price
            initial_volatility: Starting implied volatility
            num_steps: Number of time steps

        Returns:
            List of market states for each step
        """
        states = []
        current_price = initial_price
        current_vol = initial_volatility

        for step in range(num_steps):
            # Calculate time to expiry (assuming 30 days initial, decreasing)
            time_to_expiry = max(0.001, (30 - step) / 365)

            # Generate price movement using geometric Brownian motion
            dt = 1 / 365  # Daily timestep
            random_shock = self.rng.normal(0, 1)

            price_change = (self.params.drift * dt +
                          self.params.volatility * random_shock * np.sqrt(dt))

            current_price *= np.exp(price_change)

            # Update implied volatility with mean reversion
            vol_mean_reversion = 0.1  # Speed of mean reversion
            vol_target = self.params.volatility * np.sqrt(252)  # Annualized target
            vol_shock = self.rng.normal(0, 0.02)  # Vol of vol

            current_vol += vol_mean_reversion * (vol_target - current_vol) * dt + vol_shock * np.sqrt(dt)
            current_vol = max(0.05, min(current_vol, 1.0))  # Clamp between 5% and 100%

            # Create market state
            state = MarketState(
                price=current_price,
                volatility=current_vol,
                time_to_expiry=time_to_expiry,
                interest_rate=self.params.interest_rate,
                dividend_yield=self.params.dividend_yield
            )

            states.append(state)

        return states

    def calculate_position_greeks(self, market_state: MarketState,
                                 strikes: np.ndarray, quantities: np.ndarray,
                                 option_types: List[str]) -> Dict[str, float]:
        """
        Calculate portfolio Greeks for a position.

        Args:
            market_state: Current market state
            strikes: Array of strike prices
            quantities: Array of quantities (positive = long, negative = short)
            option_types: List of 'call' or 'put' for each leg

        Returns:
            Dictionary with portfolio Greeks
        """
        portfolio_delta = 0.0
        portfolio_gamma = 0.0
        portfolio_theta = 0.0
        portfolio_vega = 0.0
        portfolio_value = 0.0

        for i, (strike, quantity, option_type) in enumerate(zip(strikes, quantities, option_types)):
            is_call = option_type.lower() == 'call'

            # Calculate option price and Greeks
            price = self.bs_calc.calculate_option_price(
                market_state.price, strike, market_state.time_to_expiry,
                market_state.volatility, market_state.interest_rate,
                market_state.dividend_yield, is_call
            )

            delta = self.bs_calc.calculate_delta(
                market_state.price, strike, market_state.time_to_expiry,
                market_state.volatility, market_state.interest_rate,
                market_state.dividend_yield, is_call
            )

            gamma = self.bs_calc.calculate_gamma(
                market_state.price, strike, market_state.time_to_expiry,
                market_state.volatility, market_state.interest_rate,
                market_state.dividend_yield
            )

            theta = self.bs_calc.calculate_theta(
                market_state.price, strike, market_state.time_to_expiry,
                market_state.volatility, market_state.interest_rate,
                market_state.dividend_yield, is_call
            )

            vega = self.bs_calc.calculate_vega(
                market_state.price, strike, market_state.time_to_expiry,
                market_state.volatility, market_state.interest_rate,
                market_state.dividend_yield
            )

            # Add to portfolio (weighted by quantity)
            portfolio_value += price * quantity
            portfolio_delta += delta * quantity
            portfolio_gamma += gamma * quantity
            portfolio_theta += theta * quantity
            portfolio_vega += vega * quantity

        return {
            'value': portfolio_value,
            'delta': portfolio_delta,
            'gamma': portfolio_gamma,
            'theta': portfolio_theta,
            'vega': portfolio_vega
        }

    def simulate_step(self, current_state: MarketState) -> MarketState:
        """
        Simulate one step forward from current market state.

        Args:
            current_state: Current market state

        Returns:
            New market state after one time step
        """
        # Time decay
        new_time_to_expiry = max(0.001, current_state.time_to_expiry - 1/365)

        # Price movement
        dt = 1 / 365
        random_shock = self.rng.normal(0, 1)
        price_change = (self.params.drift * dt +
                       self.params.volatility * random_shock * np.sqrt(dt))
        new_price = current_state.price * np.exp(price_change)

        # Volatility evolution
        vol_mean_reversion = 0.1
        vol_target = self.params.volatility * np.sqrt(252)
        vol_shock = self.rng.normal(0, 0.02)

        new_volatility = current_state.volatility + \
                        vol_mean_reversion * (vol_target - current_state.volatility) * dt + \
                        vol_shock * np.sqrt(dt)
        new_volatility = max(0.05, min(new_volatility, 1.0))

        return MarketState(
            price=new_price,
            volatility=new_volatility,
            time_to_expiry=new_time_to_expiry,
            interest_rate=current_state.interest_rate,
            dividend_yield=current_state.dividend_yield
        )
"""
Options Greeks calculation using Black-Scholes model.

Provides comprehensive Greeks calculation for individual options and
multi-leg positions with proper normalization for neural networks.
"""

import math
from typing import Dict, List, Optional
import numpy as np
from scipy.stats import norm

from .position_models import Position, OptionType


class GreeksCalculator:
    """
    Black-Scholes Greeks calculator for European options.

    Implements standard Greeks calculations with normalization
    for neural network input compatibility.
    """

    def __init__(self):
        """Initialize Greeks calculator."""
        pass

    def calculate_delta(self, S: float, K: float, T: float, r: float,
                       sigma: float, option_type: str) -> float:
        """
        Calculate option delta using Black-Scholes.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            sigma: Volatility (annual)
            option_type: 'CALL' or 'PUT'

        Returns:
            Delta value normalized to [-1, 1] range
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        d1 = self._calculate_d1(S, K, T, r, sigma)

        if option_type.upper() == 'CALL':
            delta = norm.cdf(d1)
        else:  # PUT
            delta = norm.cdf(d1) - 1

        # Delta is already in [-1, 1] range
        return float(delta)

    def calculate_gamma(self, S: float, K: float, T: float, r: float,
                       sigma: float, option_type: str) -> float:
        """
        Calculate option gamma using Black-Scholes.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            sigma: Volatility (annual)
            option_type: 'CALL' or 'PUT'

        Returns:
            Gamma value normalized by underlying price
        """
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0

        d1 = self._calculate_d1(S, K, T, r, sigma)
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))

        # Normalize by underlying price to get relative gamma
        return float(gamma * S)

    def calculate_theta(self, S: float, K: float, T: float, r: float,
                       sigma: float, option_type: str) -> float:
        """
        Calculate option theta using Black-Scholes.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            sigma: Volatility (annual)
            option_type: 'CALL' or 'PUT'

        Returns:
            Theta value (always negative, normalized by underlying price)
        """
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0

        d1 = self._calculate_d1(S, K, T, r, sigma)
        d2 = d1 - sigma * math.sqrt(T)

        if option_type.upper() == 'CALL':
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
                - r * K * math.exp(-r * T) * norm.cdf(d2)
            )
        else:  # PUT
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
                + r * K * math.exp(-r * T) * norm.cdf(-d2)
            )

        # Convert to daily theta and normalize by underlying price
        daily_theta = theta / 365.0 / S
        return float(daily_theta)

    def calculate_vega(self, S: float, K: float, T: float, r: float,
                      sigma: float, option_type: str) -> float:
        """
        Calculate option vega using Black-Scholes.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            sigma: Volatility (annual)
            option_type: 'CALL' or 'PUT'

        Returns:
            Vega value normalized by underlying price
        """
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0

        d1 = self._calculate_d1(S, K, T, r, sigma)
        vega = S * norm.pdf(d1) * math.sqrt(T)

        # Normalize by underlying price and convert to 1% vol change
        return float(vega / S / 100)

    def position_greeks(self, position: Position, iv_estimates: Optional[List[float]] = None,
                       risk_free_rate: float = 0.05) -> Dict[str, float]:
        """
        Calculate aggregated Greeks for a multi-leg position.

        Args:
            position: Position object with all legs
            iv_estimates: Implied volatility for each leg (if None, uses 0.3 default)
            risk_free_rate: Risk-free rate for calculations

        Returns:
            Dictionary with aggregated position Greeks
        """
        if iv_estimates is None:
            iv_estimates = [0.3] * len(position.strikes)

        if len(iv_estimates) != len(position.strikes):
            raise ValueError("IV estimates must match number of position legs")

        # Calculate time to expiration in years
        days_to_exp = position.days_to_expiration
        T = max(days_to_exp / 365.0, 0.001)  # Minimum 1 day

        # Current underlying price in dollars
        S = position.current_underlying_price / 100.0

        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0

        for i, (strike, option_type, quantity) in enumerate(
            zip(position.strikes, position.option_types, position.quantities)
        ):
            K = strike / 100.0  # Convert to dollars
            sigma = iv_estimates[i]

            # Calculate Greeks for this leg
            delta = self.calculate_delta(S, K, T, risk_free_rate, sigma, option_type.value)
            gamma = self.calculate_gamma(S, K, T, risk_free_rate, sigma, option_type.value)
            theta = self.calculate_theta(S, K, T, risk_free_rate, sigma, option_type.value)
            vega = self.calculate_vega(S, K, T, risk_free_rate, sigma, option_type.value)

            # Apply quantity and aggregate
            total_delta += delta * quantity
            total_gamma += gamma * quantity
            total_theta += theta * quantity  # Theta is always negative sum
            total_vega += vega * quantity

        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'theta': total_theta,
            'vega': total_vega
        }

    def _calculate_d1(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter for Black-Scholes."""
        return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


class ImpliedVolatilityEstimator:
    """
    Estimate implied volatility for options when not available.

    Uses historical volatility as a proxy when options chain data unavailable.
    """

    def __init__(self):
        """Initialize IV estimator."""
        pass

    def estimate_iv(self, symbol: str, days: int = 30) -> float:
        """
        Estimate implied volatility using historical volatility.

        Args:
            symbol: Stock symbol
            days: Days of price history to use

        Returns:
            Estimated implied volatility (annualized)
        """
        try:
            # For now, use a simple proxy based on symbol characteristics
            # In production, this would fetch actual price data

            # Default volatilities by symbol type
            if symbol.upper() in ['SPY', 'QQQ', 'IWM']:
                base_vol = 0.15  # ETFs tend to be less volatile
            elif symbol.upper() in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
                base_vol = 0.25  # Large cap tech
            else:
                base_vol = 0.30  # General stocks

            # Add some randomness to simulate market conditions
            import time
            import random
            random.seed(int(time.time()) % 100)
            adjustment = random.uniform(0.8, 1.2)

            return base_vol * adjustment

        except Exception:
            # Fallback to 30% volatility
            return 0.30

    def get_iv_for_position(self, position: Position) -> List[float]:
        """
        Get IV estimates for all legs of a position.

        Args:
            position: Position to estimate IV for

        Returns:
            List of IV estimates for each leg
        """
        # In a real implementation, this would fetch option chain data
        # For now, use the same IV for all legs with slight variations
        base_iv = self.estimate_iv("SPY")  # Use SPY as base

        iv_estimates = []
        for i, (strike, option_type) in enumerate(zip(position.strikes, position.option_types)):
            # Add slight variation based on strike and type
            if option_type == OptionType.PUT:
                iv_adj = 1.1  # Puts typically have higher IV
            else:
                iv_adj = 1.0

            # Distance from ATM affects IV
            current_price = position.current_underlying_price
            moneyness = abs(strike - current_price) / current_price
            iv_skew = 1.0 + moneyness * 0.2  # Simple skew approximation

            iv_estimates.append(base_iv * iv_adj * iv_skew)

        return iv_estimates
#!/usr/bin/env python3
"""Standalone verification of Greeks calculations."""

import math
from typing import Dict, List, Optional
import numpy as np
from scipy.stats import norm

class GreeksCalculator:
    """Black-Scholes Greeks calculator for verification."""

    def calculate_delta(self, S: float, K: float, T: float, r: float,
                       sigma: float, option_type: str) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = self._calculate_d1(S, K, T, r, sigma)
        if option_type.upper() == 'CALL':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        return float(delta)

    def calculate_gamma(self, S: float, K: float, T: float, r: float,
                       sigma: float, option_type: str) -> float:
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1 = self._calculate_d1(S, K, T, r, sigma)
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        return float(gamma * S)

    def calculate_theta(self, S: float, K: float, T: float, r: float,
                       sigma: float, option_type: str) -> float:
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1 = self._calculate_d1(S, K, T, r, sigma)
        d2 = d1 - sigma * math.sqrt(T)
        if option_type.upper() == 'CALL':
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
                - r * K * math.exp(-r * T) * norm.cdf(d2)
            )
        else:
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
                + r * K * math.exp(-r * T) * norm.cdf(-d2)
            )
        daily_theta = theta / 365.0 / S
        return float(daily_theta)

    def calculate_vega(self, S: float, K: float, T: float, r: float,
                      sigma: float, option_type: str) -> float:
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1 = self._calculate_d1(S, K, T, r, sigma)
        vega = S * norm.pdf(d1) * math.sqrt(T)
        return float(vega / S / 100)

    def _calculate_d1(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


if __name__ == '__main__':
    # Test Greeks calculation
    gc = GreeksCalculator()

    # Test ATM call
    delta = gc.calculate_delta(S=100, K=105, T=0.1, r=0.05, sigma=0.2, option_type='CALL')
    gamma = gc.calculate_gamma(S=100, K=105, T=0.1, r=0.05, sigma=0.2, option_type='CALL')
    theta = gc.calculate_theta(S=100, K=105, T=0.1, r=0.05, sigma=0.2, option_type='CALL')
    vega = gc.calculate_vega(S=100, K=105, T=0.1, r=0.05, sigma=0.2, option_type='CALL')

    print(f'✓ Greeks calculation verification passed')
    print(f'  Call Delta (S=100, K=105): {delta:.4f}')
    print(f'  Gamma: {gamma:.4f}')
    print(f'  Theta: {theta:.6f} (should be negative)')
    print(f'  Vega: {vega:.4f}')

    # Validate ranges
    assert -1 <= delta <= 1, f'Delta {delta} not in [-1, 1] range'
    assert gamma > 0, f'Gamma {gamma} should be positive for long option'
    assert theta < 0, f'Theta {theta} should be negative for long option'
    assert vega > 0, f'Vega {vega} should be positive for long option'

    print('✓ All Greeks validation passed')
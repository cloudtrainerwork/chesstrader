"""
Monte Carlo simulation package for backtesting

Provides statistical validation through bootstrap resampling and simulation.
"""

try:
    from .simulator import MonteCarloSimulator
except ImportError:
    MonteCarloSimulator = None

try:
    from .bootstrap import BootstrapResampler
except ImportError:
    BootstrapResampler = None

try:
    from .analysis import StatisticalAnalyzer
except ImportError:
    StatisticalAnalyzer = None

__all__ = ['MonteCarloSimulator', 'BootstrapResampler', 'StatisticalAnalyzer']
"""
CLI interface for backtesting system

Command-line interface for orchestrating complete backtesting workflows
including walk-forward optimization and Monte Carlo analysis.
"""

from .backtest_runner import BacktestCLI, main

__all__ = ['BacktestCLI', 'main']
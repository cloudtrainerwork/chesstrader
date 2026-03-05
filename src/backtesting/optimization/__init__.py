"""
Walk-forward optimization for strategy validation

Provides WalkForwardOptimizer for parameter validation and ParameterGrid
for parameter space definition.
"""

from .walk_forward import WalkForwardOptimizer
from .parameter_grid import ParameterGrid

__all__ = ['WalkForwardOptimizer', 'ParameterGrid']
"""
Parameter grid for optimization

Defines searchable parameter spaces for strategy optimization
with support for discrete and continuous parameters.
"""

from typing import Dict, Any, List, Union, Iterator
import itertools
import numpy as np


class ParameterGrid:
    """
    Parameter space definition for grid search optimization

    Supports discrete values and continuous ranges for strategy parameters.
    """

    def __init__(self, param_dict: Dict[str, Union[List, Dict[str, Any]]]):
        """
        Initialize parameter grid

        Args:
            param_dict: Dictionary defining parameter space
                For discrete: {'param': [value1, value2, value3]}
                For continuous: {'param': {'type': 'continuous', 'min': 0.1, 'max': 1.0, 'steps': 10}}
        """
        self.param_dict = param_dict
        self._validate_params()

    def _validate_params(self):
        """Validate parameter definitions"""
        for param_name, param_def in self.param_dict.items():
            if isinstance(param_def, list):
                # Discrete parameter - validate non-empty
                if not param_def:
                    raise ValueError(f"Discrete parameter '{param_name}' cannot be empty list")
            elif isinstance(param_def, dict):
                # Continuous parameter - validate required fields
                required_fields = ['type', 'min', 'max', 'steps']
                if param_def.get('type') == 'continuous':
                    for field in required_fields[1:]:  # Skip 'type' field
                        if field not in param_def:
                            raise ValueError(f"Continuous parameter '{param_name}' missing field: {field}")
                    if param_def['min'] >= param_def['max']:
                        raise ValueError(f"Parameter '{param_name}': min must be less than max")
                    if param_def['steps'] < 2:
                        raise ValueError(f"Parameter '{param_name}': steps must be at least 2")
            else:
                raise ValueError(f"Invalid parameter definition for '{param_name}': must be list or dict")

    def grid_search(self) -> Iterator[Dict[str, Any]]:
        """
        Generate all parameter combinations for grid search

        Yields:
            Dictionary of parameter combinations
        """
        # Convert all parameters to discrete values
        param_lists = {}

        for param_name, param_def in self.param_dict.items():
            if isinstance(param_def, list):
                # Already discrete
                param_lists[param_name] = param_def
            elif isinstance(param_def, dict) and param_def.get('type') == 'continuous':
                # Convert continuous to discrete
                param_lists[param_name] = list(
                    np.linspace(param_def['min'], param_def['max'], param_def['steps'])
                )

        # Generate Cartesian product of all parameter combinations
        param_names = list(param_lists.keys())
        param_values = list(param_lists.values())

        for combination in itertools.product(*param_values):
            yield dict(zip(param_names, combination))

    def size(self) -> int:
        """
        Calculate total number of parameter combinations

        Returns:
            Total combinations in the grid
        """
        total = 1
        for param_name, param_def in self.param_dict.items():
            if isinstance(param_def, list):
                total *= len(param_def)
            elif isinstance(param_def, dict) and param_def.get('type') == 'continuous':
                total *= param_def['steps']

        return total

    def get_param_ranges(self) -> Dict[str, Any]:
        """
        Get summary of parameter ranges

        Returns:
            Dictionary with parameter range information
        """
        ranges = {}
        for param_name, param_def in self.param_dict.items():
            if isinstance(param_def, list):
                ranges[param_name] = {
                    'type': 'discrete',
                    'values': param_def,
                    'count': len(param_def)
                }
            elif isinstance(param_def, dict) and param_def.get('type') == 'continuous':
                ranges[param_name] = {
                    'type': 'continuous',
                    'min': param_def['min'],
                    'max': param_def['max'],
                    'steps': param_def['steps']
                }

        return ranges
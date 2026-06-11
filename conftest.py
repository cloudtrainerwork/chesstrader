"""
Pytest configuration to ensure local imports resolve.

Adds the project root to sys.path so tests can import `src.*`.
"""

import sys
import builtins
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure torch is available as a builtin for tests that omit explicit imports.
builtins.torch = torch
builtins.F = F
builtins.np = np
builtins.pd = pd

# These root-level files are standalone manual/integration scripts, not pytest
# test modules. They execute code (and call sys.exit()) at import time, which
# aborts pytest collection for the entire suite. Exclude them from collection;
# run them directly (e.g. `python test_api.py`) when needed. (issue #11)
collect_ignore = [
    "test_api.py",
    "test_e2e.py",
    "test_installed_api.py",
    "test_options_simple.py",
]

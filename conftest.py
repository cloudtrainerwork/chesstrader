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

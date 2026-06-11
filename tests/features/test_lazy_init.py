"""
Regression tests for lazy feature-package imports (issue #10).

The src.features package must not eagerly import its submodules, so a broken
dependency in one module (e.g. base.py's data-layer import) cannot take down
unrelated, self-contained modules in the package.
"""

import importlib
import sys

import pytest


def _purge_features():
    for name in list(sys.modules):
        if name == "src.features" or name.startswith("src.features."):
            del sys.modules[name]


def test_importing_package_does_not_eagerly_load_base():
    """Importing the package must not pull in base (its broken dependency)."""
    _purge_features()
    importlib.import_module("src.features")
    assert "src.features.base" not in sys.modules


def test_clean_submodule_imports_without_base():
    """A self-contained submodule imports without requiring base."""
    _purge_features()
    importlib.import_module("src.features.position_models")
    assert "src.features.base" not in sys.modules


def test_unknown_attribute_raises():
    """Accessing an undefined public name still raises AttributeError."""
    _purge_features()
    features = importlib.import_module("src.features")
    with pytest.raises(AttributeError):
        features.DefinitelyNotExported

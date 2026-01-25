"""
Training module for regime detection neural networks.

Provides training infrastructure, validation metrics, and model checkpointing
for the regime detection system.
"""

from .trainer import RegimeTrainer

__all__ = ['RegimeTrainer']
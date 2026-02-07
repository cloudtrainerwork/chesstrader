"""
PPO (Proximal Policy Optimization) implementation for options trading.

This package provides the core components for training trading agents using PPO:
- PPO algorithm with clipped objective function
- Generalized Advantage Estimation (GAE)
- Actor-Critic networks optimized for trading
- Experience buffer for trajectory collection
"""

from .algorithm import PPOAlgorithm
from .gae import GAECalculator
from .networks import ActorCritic
from .buffer import PPOBuffer

__all__ = ['PPOAlgorithm', 'GAECalculator', 'ActorCritic', 'PPOBuffer']
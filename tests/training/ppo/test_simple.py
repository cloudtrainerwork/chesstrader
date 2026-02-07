"""
Simple test to verify core PPO components work without complex dependencies.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock

# Test our core imports work
from src.training.ppo.algorithm import PPOAlgorithm
from src.training.ppo.networks import ActorCritic
from src.training.ppo.buffer import PPOBuffer
from src.training.ppo.gae import GAECalculator


class TestCoreComponents(unittest.TestCase):
    """Test core PPO components."""

    def test_actor_critic_import(self):
        """Test ActorCritic can be imported and created."""
        model = ActorCritic(obs_dim=10, action_dim=4)
        self.assertIsInstance(model, torch.nn.Module)

    def test_ppo_algorithm_import(self):
        """Test PPOAlgorithm can be imported and created."""
        model = ActorCritic(obs_dim=10, action_dim=4)
        algorithm = PPOAlgorithm(model)
        self.assertIsNotNone(algorithm)

    def test_buffer_import(self):
        """Test PPOBuffer can be imported and created."""
        buffer = PPOBuffer(capacity=100, obs_dim=10)
        self.assertEqual(buffer.capacity, 100)

    def test_gae_import(self):
        """Test GAECalculator can be imported and created."""
        gae = GAECalculator(gamma=0.99, lambda_=0.95)
        self.assertEqual(gae.gamma, 0.99)


if __name__ == '__main__':
    unittest.main()
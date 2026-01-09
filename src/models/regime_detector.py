"""
Regime detection neural network for market classification.

Implements a neural network that accepts 48-dimensional feature vectors
and outputs 8-regime classification probabilities plus a confidence score.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class RegimeDetector(nn.Module):
    """
    Neural network for regime detection and classification.

    Architecture: 48 → 128 → 64 → 32 → 8 regime logits + 1 confidence score

    Features:
    - Accepts 48-dimensional feature vectors from RegimeStateVector
    - Outputs 8 regime probabilities (softmax) + 1 confidence score (sigmoid)
    - Uses batch normalization and dropout for regularization
    - Supports both CPU and GPU computation
    """

    def __init__(self, input_dim: int = 48, hidden_dims: Tuple[int, int, int] = (128, 64, 32),
                 num_regimes: int = 8, dropout_rate: float = 0.2):
        """
        Initialize the regime detector network.

        Args:
            input_dim: Input feature dimension (default: 48)
            hidden_dims: Hidden layer dimensions (default: (128, 64, 32))
            num_regimes: Number of market regimes to classify (default: 8)
            dropout_rate: Dropout rate for regularization (default: 0.2)
        """
        super(RegimeDetector, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_regimes = num_regimes
        self.dropout_rate = dropout_rate

        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])

        # Hidden layers
        self.hidden1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.hidden1_bn = nn.BatchNorm1d(hidden_dims[1])

        self.hidden2 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.hidden2_bn = nn.BatchNorm1d(hidden_dims[2])

        # Output layers
        self.regime_output = nn.Linear(hidden_dims[2], num_regimes)  # Regime classification
        self.confidence_output = nn.Linear(hidden_dims[2], 1)       # Confidence score

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization for ReLU activations
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 48)

        Returns:
            Output tensor of shape (batch_size, 9) where:
            - First 8 values are regime probabilities (softmax)
            - Last value is confidence score (sigmoid)
        """
        # Input validation
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"Expected input shape (N, {self.input_dim}), got {x.shape}")

        # Handle single batch case for BatchNorm
        single_batch = x.size(0) == 1
        if single_batch and self.training:
            # Use eval mode for BatchNorm with single samples
            self.eval()
            output = self._forward_impl(x)
            self.train()
            return output
        else:
            return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Internal forward implementation."""
        # Input layer with batch norm and activation
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Hidden layer 1
        x = self.hidden1(x)
        x = self.hidden1_bn(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Hidden layer 2
        x = self.hidden2(x)
        x = self.hidden2_bn(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Output layers
        regime_logits = self.regime_output(x)
        confidence_logits = self.confidence_output(x)

        # Apply activations
        regime_probs = F.softmax(regime_logits, dim=1)  # Regime probabilities sum to 1
        confidence_score = torch.sigmoid(confidence_logits)  # Confidence in [0, 1]

        # Combine outputs
        output = torch.cat([regime_probs, confidence_score], dim=1)

        return output

    def predict_regime(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict regime with probabilities and confidence.

        Args:
            x: Input tensor of shape (batch_size, 48)

        Returns:
            Tuple of (predicted_regimes, regime_probabilities, confidence_scores)
            - predicted_regimes: Most likely regime indices (batch_size,)
            - regime_probabilities: All regime probabilities (batch_size, 8)
            - confidence_scores: Confidence scores (batch_size, 1)
        """
        with torch.no_grad():
            output = self.forward(x)
            regime_probs = output[:, :self.num_regimes]
            confidence = output[:, self.num_regimes:]

            # Get predicted regime (argmax)
            predicted_regimes = torch.argmax(regime_probs, dim=1)

            return predicted_regimes, regime_probs, confidence

    def calculate_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate prediction uncertainty using entropy of regime probabilities.

        Higher entropy indicates higher uncertainty in regime classification.

        Args:
            x: Input tensor of shape (batch_size, 48)

        Returns:
            Uncertainty scores (batch_size, 1) - higher values = more uncertain
        """
        with torch.no_grad():
            output = self.forward(x)
            regime_probs = output[:, :self.num_regimes]

            # Calculate entropy: -sum(p * log(p))
            # Add small epsilon to avoid log(0)
            epsilon = 1e-8
            entropy = -torch.sum(regime_probs * torch.log(regime_probs + epsilon), dim=1, keepdim=True)

            # Normalize entropy to [0, 1] range
            # Maximum entropy for 8 regimes is log(8)
            max_entropy = np.log(self.num_regimes)
            normalized_entropy = entropy / max_entropy

            return normalized_entropy

    def get_device(self) -> torch.device:
        """Get the device this model is on."""
        return next(self.parameters()).device

    def to_device(self, device: torch.device) -> 'RegimeDetector':
        """Move model to specified device."""
        return self.to(device)

    def __repr__(self) -> str:
        """String representation of the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return (f"RegimeDetector(\n"
                f"  Architecture: {self.input_dim} → {' → '.join(map(str, self.hidden_dims))} → "
                f"{self.num_regimes}+1\n"
                f"  Total params: {total_params:,}\n"
                f"  Trainable params: {trainable_params:,}\n"
                f"  Dropout rate: {self.dropout_rate}\n"
                f"  Device: {self.get_device()}\n"
                f")")
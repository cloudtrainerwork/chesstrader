"""
Market encoder for integrating regime detection with spatial position representations.

Combines RegimeDetector outputs with SpatialEncoder position tensors to create
multi-channel spatial representations ready for convolutional neural network processing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

from .regime_detector import RegimeDetector
from .spatial_encoder import SpatialEncoder, SpatialConfig
from ..features.position_models import Position


@dataclass
class MarketEncoderConfig:
    """Configuration for market encoding parameters."""
    confidence_threshold: float = 0.5
    regime_smoothing: bool = True
    include_uncertainty: bool = True
    spatial_config: Optional[SpatialConfig] = None


class MarketEncoder(nn.Module):
    """
    Market encoder that integrates regime detection with spatial position data.

    Creates multi-channel spatial tensors combining:
    1. Position channel: Spatial position representation (7x6)
    2. Regime channel: Market regime probabilities mapped to spatial grid (7x6)
    3. Confidence channel: Regime confidence scores mapped to spatial grid (7x6)

    Output: (C, 7, 6) tensor where C >= 3 channels for CNN processing.
    """

    def __init__(self, regime_detector: RegimeDetector,
                 config: Optional[MarketEncoderConfig] = None):
        """
        Initialize market encoder.

        Args:
            regime_detector: Trained RegimeDetector model for market regime classification
            config: Configuration for market encoding (uses defaults if None)
        """
        super(MarketEncoder, self).__init__()

        self.regime_detector = regime_detector
        self.config = config or MarketEncoderConfig()

        # Initialize spatial encoder
        spatial_config = self.config.spatial_config or SpatialConfig()
        self.spatial_encoder = SpatialEncoder(spatial_config)

        # Channel configuration
        self.num_base_channels = 3  # Position + Regime + Confidence
        self.num_regimes = regime_detector.num_regimes

        # Regime mapping to spatial representation
        self._setup_regime_mapping()

    def _setup_regime_mapping(self) -> None:
        """Setup mapping from regime indices to spatial representations."""
        # Map regime indices to spatial patterns
        # Each regime gets a unique spatial signature for the regime channel
        self.regime_patterns = torch.zeros((self.num_regimes, 7, 6))

        # Create distinctive patterns for each regime
        for i in range(self.num_regimes):
            # Create a pattern that spreads across the spatial grid
            # Use sine/cosine patterns to create unique signatures
            pattern = torch.zeros((7, 6))

            for row in range(7):
                for col in range(6):
                    # Create unique frequency pattern for each regime
                    freq_r = (i + 1) * 0.5  # Regime-specific frequency
                    freq_c = (i + 1) * 0.3

                    value = (np.sin(row * freq_r) * np.cos(col * freq_c) +
                            np.cos(row * freq_c) * np.sin(col * freq_r))
                    pattern[row, col] = value

            # Normalize pattern
            pattern = torch.tanh(pattern)
            self.regime_patterns[i] = pattern

        # Register as buffer so it moves with model device
        self.register_buffer('regime_spatial_patterns', self.regime_patterns)

    def forward(self, position: Position,
                market_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - encode market state with position.

        Args:
            position: Position object to encode spatially
            market_features: Market feature tensor for regime detection (48-dim)

        Returns:
            Multi-channel spatial tensor of shape (C, 7, 6) where C >= 3
        """
        return self.encode_market_state(position, market_features)

    def encode_market_state(self, position: Position,
                           market_features: torch.Tensor) -> torch.Tensor:
        """
        Encode market state combining position and regime information.

        Creates multi-channel spatial representation where:
        - Channel 0: Spatial position encoding
        - Channel 1: Market regime spatial pattern
        - Channel 2: Regime confidence spatial map
        - Channel 3+: Optional uncertainty and metadata channels

        Args:
            position: Position object to encode
            market_features: Market feature vector (batch_size, 48) or (48,)

        Returns:
            Tensor of shape (C, 7, 6) with market-integrated spatial encoding
        """
        # Handle single sample vs batch
        if market_features.dim() == 1:
            market_features = market_features.unsqueeze(0)  # Add batch dimension
            single_sample = True
        else:
            single_sample = False

        batch_size = market_features.shape[0]

        # 1. Get spatial position encoding
        position_tensor = self.spatial_encoder.position_to_spatial(position)
        if single_sample:
            position_channel = position_tensor.unsqueeze(0)  # (1, 7, 6)
        else:
            # Repeat for batch
            position_channel = position_tensor.unsqueeze(0).expand(batch_size, -1, -1)

        # 2. Get regime predictions
        regime_output = self.regime_detector(market_features)
        regime_probs = regime_output[:, :self.num_regimes]  # (batch_size, 8)
        confidence_scores = regime_output[:, self.num_regimes:]  # (batch_size, 1)

        # Apply confidence thresholding if configured
        if hasattr(self.config, 'confidence_threshold'):
            low_confidence_mask = confidence_scores < self.config.confidence_threshold
            regime_probs = self._apply_confidence_threshold(regime_probs, low_confidence_mask)

        # 3. Create regime spatial channel
        regime_channel = self._create_regime_spatial_channel(regime_probs, batch_size)

        # 4. Create confidence spatial channel
        confidence_channel = self._create_confidence_spatial_channel(confidence_scores, batch_size)

        # 5. Combine all channels
        channels = [
            position_channel,  # (batch_size, 7, 6)
            regime_channel,    # (batch_size, 7, 6)
            confidence_channel  # (batch_size, 7, 6)
        ]

        # 6. Add optional uncertainty channel
        if self.config.include_uncertainty:
            uncertainty_channel = self._create_uncertainty_channel(regime_probs, batch_size)
            channels.append(uncertainty_channel)

        # Stack channels: (batch_size, C, 7, 6)
        multi_channel_tensor = torch.stack(channels, dim=1)

        # Return single sample if input was single sample
        if single_sample:
            return multi_channel_tensor.squeeze(0)  # (C, 7, 6)
        else:
            return multi_channel_tensor  # (batch_size, C, 7, 6)

    def _apply_confidence_threshold(self, regime_probs: torch.Tensor,
                                  low_confidence_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply confidence thresholding to regime probabilities.

        For low-confidence predictions, smooth towards uniform distribution.

        Args:
            regime_probs: Regime probability tensor (batch_size, 8)
            low_confidence_mask: Boolean mask for low confidence samples (batch_size, 1)

        Returns:
            Adjusted regime probabilities
        """
        adjusted_probs = regime_probs.clone()

        # For low confidence samples, blend with uniform distribution
        uniform_dist = torch.ones_like(regime_probs) / self.num_regimes
        blend_factor = 0.5  # 50% blend with uniform for low confidence

        low_confidence_indices = low_confidence_mask.squeeze(1)
        if torch.any(low_confidence_indices):
            adjusted_probs[low_confidence_indices] = (
                blend_factor * uniform_dist[low_confidence_indices] +
                (1 - blend_factor) * regime_probs[low_confidence_indices]
            )

        return adjusted_probs

    def _create_regime_spatial_channel(self, regime_probs: torch.Tensor,
                                     batch_size: int) -> torch.Tensor:
        """
        Create spatial channel encoding regime information.

        Args:
            regime_probs: Regime probabilities (batch_size, 8)
            batch_size: Number of samples in batch

        Returns:
            Regime spatial channel (batch_size, 7, 6)
        """
        regime_channel = torch.zeros((batch_size, 7, 6), device=regime_probs.device)

        for i in range(batch_size):
            # Get regime probabilities for this sample
            probs = regime_probs[i]  # (8,)

            # Create weighted combination of regime patterns
            spatial_pattern = torch.zeros((7, 6), device=regime_probs.device)

            for regime_idx in range(self.num_regimes):
                prob = probs[regime_idx]
                pattern = self.regime_spatial_patterns[regime_idx]  # (7, 6)
                spatial_pattern += prob * pattern

            # Optional smoothing
            if self.config.regime_smoothing:
                spatial_pattern = self._apply_spatial_smoothing(spatial_pattern)

            regime_channel[i] = spatial_pattern

        return regime_channel

    def _create_confidence_spatial_channel(self, confidence_scores: torch.Tensor,
                                         batch_size: int) -> torch.Tensor:
        """
        Create spatial channel encoding confidence information.

        Args:
            confidence_scores: Confidence scores (batch_size, 1)
            batch_size: Number of samples in batch

        Returns:
            Confidence spatial channel (batch_size, 7, 6)
        """
        confidence_channel = torch.zeros((batch_size, 7, 6), device=confidence_scores.device)

        for i in range(batch_size):
            confidence = confidence_scores[i, 0]  # Scalar confidence

            # Create confidence pattern - higher confidence = stronger pattern
            # Use a radial pattern centered in the spatial grid
            pattern = torch.zeros((7, 6))

            center_r, center_c = 3, 2.5  # Center of 7x6 grid

            for row in range(7):
                for col in range(6):
                    # Distance from center
                    dist = np.sqrt((row - center_r)**2 + (col - center_c)**2)
                    max_dist = np.sqrt(center_r**2 + center_c**2)

                    # Radial confidence pattern
                    normalized_dist = dist / max_dist
                    radial_value = confidence * (1 - normalized_dist) * np.exp(-normalized_dist)

                    pattern[row, col] = radial_value

            confidence_channel[i] = pattern

        return confidence_channel

    def _create_uncertainty_channel(self, regime_probs: torch.Tensor,
                                  batch_size: int) -> torch.Tensor:
        """
        Create spatial channel encoding prediction uncertainty.

        Uses entropy of regime probabilities as uncertainty measure.

        Args:
            regime_probs: Regime probabilities (batch_size, 8)
            batch_size: Number of samples in batch

        Returns:
            Uncertainty spatial channel (batch_size, 7, 6)
        """
        uncertainty_channel = torch.zeros((batch_size, 7, 6), device=regime_probs.device)

        for i in range(batch_size):
            probs = regime_probs[i]  # (8,)

            # Calculate entropy (uncertainty)
            epsilon = 1e-8
            entropy = -torch.sum(probs * torch.log(probs + epsilon))
            max_entropy = np.log(self.num_regimes)
            normalized_uncertainty = entropy / max_entropy

            # Create uncertainty pattern - higher uncertainty = more diffuse pattern
            pattern = torch.zeros((7, 6))

            # Use noise-like pattern for uncertainty
            for row in range(7):
                for col in range(6):
                    # Create pseudo-random pattern based on position
                    seed_val = (row * 6 + col + i) % 23  # Deterministic "randomness"
                    noise_val = np.sin(seed_val * 2.7) * np.cos(seed_val * 1.3)
                    pattern[row, col] = normalized_uncertainty * noise_val

            uncertainty_channel[i] = pattern

        return uncertainty_channel

    def _apply_spatial_smoothing(self, spatial_pattern: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial smoothing to regime patterns.

        Args:
            spatial_pattern: Input pattern (7, 6)

        Returns:
            Smoothed pattern (7, 6)
        """
        # Simple 3x3 average filter
        smoothed = spatial_pattern.clone()
        original = spatial_pattern.clone()

        for row in range(1, 6):  # Skip edges
            for col in range(1, 5):
                # 3x3 average
                neighborhood = original[row-1:row+2, col-1:col+2]
                smoothed[row, col] = torch.mean(neighborhood)

        return smoothed

    def encode_batch_market_state(self, positions: List[Position],
                                market_features_batch: torch.Tensor) -> torch.Tensor:
        """
        Encode batch of positions with market states.

        Args:
            positions: List of Position objects
            market_features_batch: Market features (batch_size, 48)

        Returns:
            Batch tensor of shape (batch_size, C, 7, 6)
        """
        batch_size = len(positions)
        if batch_size != market_features_batch.shape[0]:
            raise ValueError(
                f"Position count ({batch_size}) doesn't match market features batch size "
                f"({market_features_batch.shape[0]})"
            )

        # Process all market features at once
        regime_output = self.regime_detector(market_features_batch)
        regime_probs = regime_output[:, :self.num_regimes]
        confidence_scores = regime_output[:, self.num_regimes:]

        # Apply confidence thresholding
        if hasattr(self.config, 'confidence_threshold'):
            low_confidence_mask = confidence_scores < self.config.confidence_threshold
            regime_probs = self._apply_confidence_threshold(regime_probs, low_confidence_mask)

        # Create spatial tensors for all positions
        position_channels = []
        for position in positions:
            pos_tensor = self.spatial_encoder.position_to_spatial(position)
            position_channels.append(pos_tensor)

        position_batch = torch.stack(position_channels)  # (batch_size, 7, 6)

        # Create regime and confidence channels
        regime_channel = self._create_regime_spatial_channel(regime_probs, batch_size)
        confidence_channel = self._create_confidence_spatial_channel(confidence_scores, batch_size)

        # Combine channels
        channels = [position_batch, regime_channel, confidence_channel]

        # Add uncertainty channel if configured
        if self.config.include_uncertainty:
            uncertainty_channel = self._create_uncertainty_channel(regime_probs, batch_size)
            channels.append(uncertainty_channel)

        # Stack to create final tensor: (batch_size, C, 7, 6)
        batch_tensor = torch.stack(channels, dim=1)

        return batch_tensor

    def get_channel_info(self) -> Dict[str, int]:
        """
        Get information about output channels.

        Returns:
            Dictionary mapping channel names to indices
        """
        info = {
            'position': 0,
            'regime': 1,
            'confidence': 2
        }

        if self.config.include_uncertainty:
            info['uncertainty'] = 3

        info['total_channels'] = len(info) - 1  # Exclude total_channels key itself

        return info

    def visualize_market_encoding(self, position: Position,
                                market_features: torch.Tensor) -> str:
        """
        Create text visualization of market encoding for debugging.

        Args:
            position: Position to encode
            market_features: Market features (48-dim)

        Returns:
            String representation of the multi-channel encoding
        """
        encoded_tensor = self.encode_market_state(position, market_features)
        C, H, W = encoded_tensor.shape

        lines = []
        lines.append(f"Market Encoding Visualization ({C} channels, {H}x{W}):")
        lines.append("=" * 60)

        channel_names = ['Position', 'Regime', 'Confidence']
        if self.config.include_uncertainty:
            channel_names.append('Uncertainty')

        for c in range(C):
            channel_name = channel_names[c] if c < len(channel_names) else f'Channel_{c}'
            lines.append(f"\nChannel {c}: {channel_name}")
            lines.append("-" * 40)

            channel_tensor = encoded_tensor[c]
            for row in range(H):
                row_str = ""
                for col in range(W):
                    value = channel_tensor[row, col].item()
                    row_str += f"{value:7.3f} "
                lines.append(row_str)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation of MarketEncoder."""
        channel_info = self.get_channel_info()
        total_channels = channel_info['total_channels']

        return (
            f"MarketEncoder(\n"
            f"  Output shape: ({total_channels}, 7, 6)\n"
            f"  Channels: {list(channel_info.keys())[:-1]}\n"  # Exclude total_channels
            f"  Regime detector: {self.num_regimes} regimes\n"
            f"  Confidence threshold: {self.config.confidence_threshold}\n"
            f"  Include uncertainty: {self.config.include_uncertainty}\n"
            f"  Regime smoothing: {self.config.regime_smoothing}\n"
            f")"
        )
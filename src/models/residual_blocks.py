"""
Residual blocks for chess-inspired deep feature extraction.

Implements ResidualBlock and ResidualStack classes for deep spatial feature learning
with chess-inspired architecture suitable for 7x6 spatial tensors from options positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualBlock(nn.Module):
    """
    Chess-inspired residual block for spatial feature extraction.

    Uses conv layers with 3x3 and 1x1 kernels for spatial and pointwise convolutions,
    batch normalization, ReLU activation, and residual connections with identity mapping.
    Designed to preserve spatial dimensions (7x6) while increasing feature depth.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 padding: int = 1):
        """
        Initialize residual block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Convolution stride (default: 1)
            padding: Convolution padding (default: 1)
        """
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # Main convolution path
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 pointwise convolution for feature mixing
        self.pointwise_conv = nn.Conv2d(out_channels, out_channels,
                                       kernel_size=1, stride=1, padding=0, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)

        # Skip connection adaptation
        self.skip_connection = None
        if stride != 1 or in_channels != out_channels:
            # Need to adapt dimensions for residual connection
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # Initialize weights using He initialization for ReLU
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using He initialization for ReLU activations."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        # Store input for residual connection
        identity = x

        # Main convolution path
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Pointwise convolution for feature mixing
        out = self.pointwise_conv(out)
        out = self.pointwise_bn(out)

        # Adapt skip connection if needed
        if self.skip_connection is not None:
            identity = self.skip_connection(identity)

        # Add residual connection
        out = out + identity

        # Final activation
        out = F.relu(out)

        return out

    def __repr__(self) -> str:
        """String representation of ResidualBlock."""
        return (f"ResidualBlock({self.in_channels}, {self.out_channels}, "
                f"stride={self.stride})")


class ResidualStack(nn.Module):
    """
    Stack of residual blocks for deep feature extraction.

    Chains multiple ResidualBlock instances to create deep feature extraction
    network suitable for chess-inspired spatial analysis of options positions.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_blocks: int = 4,
                 first_stride: int = 1):
        """
        Initialize residual stack.

        Args:
            in_channels: Number of input channels
            hidden_channels: Number of hidden channels in residual blocks
            out_channels: Number of output channels
            num_blocks: Number of residual blocks (default: 4)
            first_stride: Stride for first block (default: 1)
        """
        super(ResidualStack, self).__init__()

        if num_blocks < 1:
            raise ValueError("Number of blocks must be at least 1")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks

        # Build stack of residual blocks
        self.blocks = nn.ModuleList()

        # First block may have different input/output channels and stride
        self.blocks.append(
            ResidualBlock(in_channels, hidden_channels, stride=first_stride)
        )

        # Middle blocks (hidden -> hidden)
        for _ in range(num_blocks - 2):
            self.blocks.append(
                ResidualBlock(hidden_channels, hidden_channels, stride=1)
            )

        # Last block (hidden -> output)
        if num_blocks > 1:
            self.blocks.append(
                ResidualBlock(hidden_channels, out_channels, stride=1)
            )
        else:
            # Single block case - adjust first block output channels
            self.blocks[0] = ResidualBlock(in_channels, out_channels, stride=first_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual stack.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Output tensor after processing through all residual blocks
        """
        # Pass through each residual block sequentially
        for block in self.blocks:
            x = block(x)

        return x

    def get_feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Get intermediate feature maps from each residual block.

        Useful for visualization and analysis of learned features.

        Args:
            x: Input tensor

        Returns:
            List of feature maps from each block
        """
        feature_maps = []

        for block in self.blocks:
            x = block(x)
            feature_maps.append(x.clone())

        return feature_maps

    def count_parameters(self) -> int:
        """Count total number of parameters in the residual stack."""
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        """String representation of ResidualStack."""
        total_params = self.count_parameters()
        return (f"ResidualStack(\n"
                f"  Architecture: {self.in_channels} → {self.hidden_channels} → {self.out_channels}\n"
                f"  Num blocks: {self.num_blocks}\n"
                f"  Total parameters: {total_params:,}\n"
                f"  Blocks: {len(self.blocks)}\n"
                f")")


class ChessInspiredFeatureExtractor(nn.Module):
    """
    Complete chess-inspired feature extractor using residual blocks.

    Takes multi-channel spatial tensors (e.g., from MarketEncoder) and extracts
    deep spatial features using residual blocks optimized for 7x6 board dimensions.
    """

    def __init__(self,
                 input_channels: int,
                 base_channels: int = 32,
                 num_stages: int = 3,
                 blocks_per_stage: int = 2,
                 output_features: Optional[int] = None):
        """
        Initialize chess-inspired feature extractor.

        Args:
            input_channels: Number of input channels (e.g., 3-4 from MarketEncoder)
            base_channels: Base number of channels (doubled each stage)
            num_stages: Number of stages (each with different channel depth)
            blocks_per_stage: Number of residual blocks per stage
            output_features: Optional final feature dimension (if None, uses spatial features)
        """
        super(ChessInspiredFeatureExtractor, self).__init__()

        self.input_channels = input_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.blocks_per_stage = blocks_per_stage
        self.output_features = output_features

        # Build stages with increasing channel depth
        self.stages = nn.ModuleList()
        current_channels = input_channels

        for stage_idx in range(num_stages):
            # Calculate output channels for this stage (exponential growth)
            stage_out_channels = base_channels * (2 ** stage_idx)

            # Create residual stack for this stage
            stage = ResidualStack(
                in_channels=current_channels,
                hidden_channels=stage_out_channels,
                out_channels=stage_out_channels,
                num_blocks=blocks_per_stage,
                first_stride=1  # Preserve spatial dimensions
            )

            self.stages.append(stage)
            current_channels = stage_out_channels

        # Optional global feature extraction
        if output_features is not None:
            # Global average pooling + fully connected
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(current_channels, output_features)
        else:
            self.global_pool = None
            self.fc = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extractor.

        Args:
            x: Input tensor of shape (batch_size, input_channels, 7, 6)

        Returns:
            Extracted features (spatial or global based on configuration)
        """
        # Validate input dimensions
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (batch, channels, height, width), got {x.dim()}D")

        if x.size(2) != 7 or x.size(3) != 6:
            raise ValueError(f"Expected spatial dimensions (7, 6), got ({x.size(2)}, {x.size(3)})")

        # Pass through each stage
        for stage in self.stages:
            x = stage(x)

        # Optional global feature extraction
        if self.global_pool is not None and self.fc is not None:
            # Global average pooling
            x = self.global_pool(x)  # (batch, channels, 1, 1)
            x = x.view(x.size(0), -1)  # (batch, channels)
            x = self.fc(x)  # (batch, output_features)

        return x

    def __repr__(self) -> str:
        """String representation of ChessInspiredFeatureExtractor."""
        total_params = sum(p.numel() for p in self.parameters())
        return (f"ChessInspiredFeatureExtractor(\n"
                f"  Input channels: {self.input_channels}\n"
                f"  Base channels: {self.base_channels}\n"
                f"  Stages: {self.num_stages} (blocks per stage: {self.blocks_per_stage})\n"
                f"  Output features: {self.output_features or 'Spatial'}\n"
                f"  Total parameters: {total_params:,}\n"
                f")")
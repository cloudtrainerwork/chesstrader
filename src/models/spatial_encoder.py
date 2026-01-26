"""
Spatial encoder for converting options positions to chess-inspired spatial representations.

Transforms options strategy positions into 7x6 spatial tensors where:
- 7 rows represent price levels (zones from PositionZones enum)
- 6 columns represent option legs/strikes (up to 6)
- Cell values encode option type, quantity, and Greeks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from ..features.position_models import Position, PositionZones, StrategyType, OptionType


# Constants for spatial dimensions
BOARD_ROWS = 7  # Price zones from PositionZones
BOARD_COLS = 6  # Maximum option legs supported


@dataclass
class SpatialConfig:
    """Configuration for spatial encoding parameters."""
    board_rows: int = BOARD_ROWS
    board_cols: int = BOARD_COLS
    encoding_scale: float = 1.0
    normalize_output: bool = True


class SpatialEncoder(nn.Module):
    """
    Spatial encoder that converts options positions to 7x6 spatial tensors.

    Maps options strategies to chess-board-like representations where:
    - Rows represent price levels (7 zones from PositionZones)
    - Columns represent option legs/strikes (up to 6)
    - Cell values encode option type, quantity, and Greeks

    This enables transfer learning from chess AI architectures for options strategy analysis.
    """

    def __init__(self, config: Optional[SpatialConfig] = None):
        """
        Initialize spatial encoder.

        Args:
            config: Configuration for spatial encoding (uses defaults if None)
        """
        super(SpatialEncoder, self).__init__()

        self.config = config or SpatialConfig()
        self.board_rows = self.config.board_rows
        self.board_cols = self.config.board_cols

        # Price zone mapping (from PositionZones enum)
        self.zone_to_row = {
            PositionZones.DEEP_LOSS: 0,    # Bottom row for maximum loss
            PositionZones.LOSS: 1,
            PositionZones.WARNING: 2,
            PositionZones.SAFE: 3,         # Center row for breakeven/safe zone
            PositionZones.PROFIT: 4,
            PositionZones.HIGH_PROFIT: 5,
            PositionZones.MAX_PROFIT: 6    # Top row for maximum profit
        }

        # Option type encoding values
        self.option_encoding = {
            OptionType.CALL: 1.0,
            OptionType.PUT: -1.0
        }

    def forward(self, position: Position) -> torch.Tensor:
        """
        Forward pass - convert position to spatial tensor.

        Args:
            position: Position object to encode

        Returns:
            Spatial tensor of shape (7, 6)
        """
        return self.position_to_spatial(position)

    def position_to_spatial(self, position: Position) -> torch.Tensor:
        """
        Convert Position object to 7x6 spatial tensor representation.

        Maps position components to spatial board where:
        - Rows represent price zones relative to current underlying price
        - Columns represent option legs (strikes sorted by value)
        - Cell values combine option type, quantity, and relative position

        Args:
            position: Position object to convert

        Returns:
            Tensor of shape (7, 6) representing spatial position encoding
        """
        # Initialize empty spatial tensor
        spatial_tensor = torch.zeros((self.board_rows, self.board_cols), dtype=torch.float32)

        # Get current price zone for reference
        current_zone = position.calculate_price_zone()
        current_zone_row = self.zone_to_row[current_zone]

        # Sort strikes with their associated data for consistent column mapping
        strike_data = list(zip(
            position.strikes,
            position.option_types,
            position.quantities,
            position.entry_prices,
            position.current_prices
        ))
        strike_data.sort(key=lambda x: x[0])  # Sort by strike price

        # Limit to maximum supported legs
        max_legs = min(len(strike_data), self.board_cols)

        for col, (strike, option_type, quantity, entry_price, current_price) in enumerate(strike_data[:max_legs]):
            # Calculate price zone for this strike relative to current underlying
            strike_zone_row = self._calculate_strike_zone(
                strike,
                position.current_underlying_price,
                current_zone_row
            )

            # Ensure row is within bounds
            row = max(0, min(strike_zone_row, self.board_rows - 1))

            # Calculate cell encoding value
            cell_value = self._encode_position_data(
                option_type, quantity, entry_price, current_price, strike, position.current_underlying_price
            )

            # Place encoded value in spatial tensor
            spatial_tensor[row, col] = cell_value

            # Add quantity information in adjacent rows if significant position
            if abs(quantity) > 1:
                # Spread quantity influence to adjacent zones
                self._add_quantity_influence(spatial_tensor, row, col, quantity, cell_value)

        # Validate output dimensions
        self.validate_spatial_dimensions(spatial_tensor)

        if self.config.normalize_output:
            spatial_tensor = self._normalize_tensor(spatial_tensor)

        return spatial_tensor

    def _calculate_strike_zone(self, strike: int, underlying_price: int, current_zone_row: int) -> int:
        """
        Calculate which row (price zone) a strike should map to.

        Args:
            strike: Strike price in cents
            underlying_price: Current underlying price in cents
            current_zone_row: Row index of current price zone

        Returns:
            Row index (0-6) for the strike's spatial position
        """
        # Calculate relative distance from strike to underlying
        price_diff_pct = (strike - underlying_price) / underlying_price

        # Map percentage difference to row offset from current zone
        # Use tanh to bound the mapping
        row_offset = int(np.tanh(price_diff_pct * 3) * 2)  # Scale and bound offset

        # Calculate target row
        target_row = current_zone_row + row_offset

        # Ensure within bounds
        return max(0, min(target_row, self.board_rows - 1))

    def _encode_position_data(self, option_type: OptionType, quantity: int,
                             entry_price: int, current_price: int,
                             strike: int, underlying_price: int) -> float:
        """
        Encode position data into a single cell value.

        Combines option type, quantity, and position P/L into single encoding.

        Args:
            option_type: CALL or PUT
            quantity: Number of contracts (negative for short)
            entry_price: Entry price in cents
            current_price: Current market price in cents
            strike: Strike price in cents
            underlying_price: Current underlying price in cents

        Returns:
            Encoded cell value combining all position information
        """
        # Base encoding from option type
        base_value = self.option_encoding[option_type]

        # Scale by normalized quantity (tanh to bound)
        quantity_scale = np.tanh(quantity / 10.0)  # Normalize around 10 contracts

        # Calculate position P/L factor
        if current_price != 0 and entry_price != 0:
            pnl_factor = (current_price - entry_price) / entry_price
            pnl_scale = np.tanh(pnl_factor)
        else:
            pnl_scale = 0.0

        # Calculate moneyness factor
        moneyness = (underlying_price - strike) / strike
        moneyness_scale = np.tanh(moneyness * 2)  # Scale moneyness impact

        # Combine all factors
        # Format: base_type * (1 + quantity_influence + pnl_influence + moneyness_influence)
        encoded_value = base_value * (1.0 + 0.3 * quantity_scale + 0.2 * pnl_scale + 0.1 * moneyness_scale)

        # Apply encoding scale
        encoded_value *= self.config.encoding_scale

        return float(encoded_value)

    def _add_quantity_influence(self, tensor: torch.Tensor, row: int, col: int,
                               quantity: int, base_value: float) -> None:
        """
        Add quantity influence to adjacent cells for large positions.

        Large positions influence neighboring zones to represent their market impact.

        Args:
            tensor: Spatial tensor to modify in-place
            row: Center row position
            col: Column position
            quantity: Position quantity
            base_value: Base encoded value
        """
        if abs(quantity) <= 1:
            return

        # Calculate influence strength
        influence_strength = min(abs(quantity) / 20.0, 0.5)  # Cap at 50% influence
        influence_value = base_value * influence_strength

        # Add influence to adjacent rows
        for offset in [-1, 1]:
            adj_row = row + offset
            if 0 <= adj_row < self.board_rows:
                # Add influence, don't replace existing values
                tensor[adj_row, col] += influence_value * 0.5  # Reduced influence for adjacent

    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize spatial tensor for consistent neural network input.

        Args:
            tensor: Input tensor to normalize

        Returns:
            Normalized tensor with values in reasonable range
        """
        # Clamp extreme values
        tensor = torch.clamp(tensor, -5.0, 5.0)

        # Apply tanh normalization to bound in [-1, 1]
        normalized = torch.tanh(tensor)

        return normalized

    def validate_spatial_dimensions(self, tensor: torch.Tensor) -> None:
        """
        Validate that spatial tensor has correct dimensions.

        Args:
            tensor: Tensor to validate

        Raises:
            ValueError: If tensor dimensions are incorrect
        """
        expected_shape = (self.board_rows, self.board_cols)

        if tensor.shape != expected_shape:
            raise ValueError(
                f"Spatial tensor must have shape {expected_shape}, got {tensor.shape}"
            )

        # Check for invalid values
        if torch.any(torch.isnan(tensor)):
            raise ValueError("Spatial tensor contains NaN values")

        if torch.any(torch.isinf(tensor)):
            raise ValueError("Spatial tensor contains infinite values")

    def encode_batch(self, positions: List[Position]) -> torch.Tensor:
        """
        Encode multiple positions into batch of spatial tensors.

        Args:
            positions: List of Position objects to encode

        Returns:
            Tensor of shape (batch_size, 7, 6) with spatial encodings
        """
        batch_size = len(positions)
        batch_tensor = torch.zeros((batch_size, self.board_rows, self.board_cols))

        for i, position in enumerate(positions):
            batch_tensor[i] = self.position_to_spatial(position)

        return batch_tensor

    def decode_spatial_info(self, spatial_tensor: torch.Tensor) -> Dict[str, any]:
        """
        Extract interpretable information from spatial tensor for debugging.

        Args:
            spatial_tensor: Spatial tensor to analyze

        Returns:
            Dictionary with spatial tensor analysis
        """
        info = {
            'shape': spatial_tensor.shape,
            'non_zero_cells': int(torch.count_nonzero(spatial_tensor)),
            'value_range': {
                'min': float(torch.min(spatial_tensor)),
                'max': float(torch.max(spatial_tensor)),
                'mean': float(torch.mean(spatial_tensor))
            },
            'active_zones': [],
            'active_columns': []
        }

        # Find active zones (rows with non-zero values)
        for row in range(self.board_rows):
            if torch.any(spatial_tensor[row, :] != 0):
                zone_name = [k for k, v in self.zone_to_row.items() if v == row][0]
                info['active_zones'].append({'row': row, 'zone': zone_name.name})

        # Find active columns (legs with positions)
        for col in range(self.board_cols):
            if torch.any(spatial_tensor[:, col] != 0):
                info['active_columns'].append(col)

        return info

    def visualize_spatial_tensor(self, spatial_tensor: torch.Tensor) -> str:
        """
        Create text visualization of spatial tensor for debugging.

        Args:
            spatial_tensor: Tensor to visualize

        Returns:
            String representation of the spatial tensor
        """
        lines = []
        lines.append("Spatial Tensor Visualization (7x6):")
        lines.append("Rows: Price Zones | Cols: Option Legs")
        lines.append("-" * 50)

        # Row labels (price zones)
        zone_labels = [
            "MAX_PROFIT ", "HIGH_PROFIT", "PROFIT     ", "SAFE       ",
            "WARNING    ", "LOSS       ", "DEEP_LOSS  "
        ]

        for row in range(self.board_rows):
            row_str = f"{zone_labels[row]} |"
            for col in range(self.board_cols):
                value = spatial_tensor[row, col].item()
                row_str += f"{value:8.3f}"
            lines.append(row_str)

        lines.append("-" * 50)
        lines.append("Columns represent option legs (sorted by strike)")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation of SpatialEncoder."""
        return (
            f"SpatialEncoder(\n"
            f"  Board dimensions: {self.board_rows}x{self.board_cols}\n"
            f"  Encoding scale: {self.config.encoding_scale}\n"
            f"  Normalize output: {self.config.normalize_output}\n"
            f"  Supported strategies: {len(StrategyType)} types\n"
            f")"
        )
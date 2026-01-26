"""
Test suite for SpatialEncoder functionality.

Tests spatial encoding of options positions into 7x6 tensors,
covering all 16 strategy types and edge cases.
"""

import pytest
import torch
import numpy as np
from datetime import datetime, timedelta
from typing import List

from src.models.spatial_encoder import SpatialEncoder, SpatialConfig, BOARD_ROWS, BOARD_COLS
from src.features.position_models import (
    Position, StrategyType, OptionType, PositionZones
)


class TestSpatialEncoder:
    """Test suite for SpatialEncoder class."""

    @pytest.fixture
    def encoder(self):
        """Create SpatialEncoder instance for testing."""
        return SpatialEncoder()

    @pytest.fixture
    def custom_encoder(self):
        """Create SpatialEncoder with custom configuration."""
        config = SpatialConfig(
            board_rows=7,
            board_cols=6,
            encoding_scale=2.0,
            normalize_output=False
        )
        return SpatialEncoder(config)

    @pytest.fixture
    def sample_position(self):
        """Create sample position for testing."""
        return Position(
            strategy_type=StrategyType.BULL_CALL_SPREAD,
            entry_date=datetime.now() - timedelta(days=30),
            expiration_date=datetime.now() + timedelta(days=15),
            strikes=[9500, 10000],  # $95, $100 strikes in cents
            option_types=[OptionType.CALL, OptionType.CALL],
            quantities=[1, -1],  # Long lower strike, short higher strike
            entry_prices=[500, 300],  # $5.00, $3.00 in cents
            current_prices=[300, 100],  # $3.00, $1.00 in cents
            underlying_price_at_entry=9750,  # $97.50
            current_underlying_price=9800   # $98.00
        )

    @pytest.fixture
    def all_strategy_positions(self):
        """Create positions for all 16 strategy types."""
        base_date = datetime.now()
        positions = []

        # Single leg strategies
        positions.append(Position(
            strategy_type=StrategyType.LONG_CALL,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[10000],
            option_types=[OptionType.CALL],
            quantities=[1],
            entry_prices=[500],
            current_prices=[300],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        ))

        positions.append(Position(
            strategy_type=StrategyType.SHORT_CALL,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[10000],
            option_types=[OptionType.CALL],
            quantities=[-1],
            entry_prices=[500],
            current_prices=[300],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        ))

        positions.append(Position(
            strategy_type=StrategyType.LONG_PUT,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[9500],
            option_types=[OptionType.PUT],
            quantities=[1],
            entry_prices=[400],
            current_prices=[200],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        ))

        positions.append(Position(
            strategy_type=StrategyType.SHORT_PUT,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[9500],
            option_types=[OptionType.PUT],
            quantities=[-1],
            entry_prices=[400],
            current_prices=[200],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        ))

        # Vertical spreads
        positions.append(Position(
            strategy_type=StrategyType.BULL_CALL_SPREAD,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[9500, 10000],
            option_types=[OptionType.CALL, OptionType.CALL],
            quantities=[1, -1],
            entry_prices=[600, 400],
            current_prices=[400, 200],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        ))

        positions.append(Position(
            strategy_type=StrategyType.BEAR_CALL_SPREAD,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[9500, 10000],
            option_types=[OptionType.CALL, OptionType.CALL],
            quantities=[-1, 1],
            entry_prices=[600, 400],
            current_prices=[400, 200],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        ))

        positions.append(Position(
            strategy_type=StrategyType.BULL_PUT_SPREAD,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[9000, 9500],
            option_types=[OptionType.PUT, OptionType.PUT],
            quantities=[-1, 1],
            entry_prices=[200, 400],
            current_prices=[100, 200],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        ))

        positions.append(Position(
            strategy_type=StrategyType.BEAR_PUT_SPREAD,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[9000, 9500],
            option_types=[OptionType.PUT, OptionType.PUT],
            quantities=[1, -1],
            entry_prices=[200, 400],
            current_prices=[100, 200],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        ))

        # Horizontal spreads
        positions.append(Position(
            strategy_type=StrategyType.CALENDAR_CALL,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[10000, 10000],
            option_types=[OptionType.CALL, OptionType.CALL],
            quantities=[-1, 1],  # Short near, long far
            entry_prices=[500, 700],
            current_prices=[300, 500],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        ))

        positions.append(Position(
            strategy_type=StrategyType.CALENDAR_PUT,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[9500, 9500],
            option_types=[OptionType.PUT, OptionType.PUT],
            quantities=[-1, 1],  # Short near, long far
            entry_prices=[300, 500],
            current_prices=[200, 400],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        ))

        # Volatility strategies
        positions.append(Position(
            strategy_type=StrategyType.LONG_STRADDLE,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[9800, 9800],
            option_types=[OptionType.CALL, OptionType.PUT],
            quantities=[1, 1],
            entry_prices=[400, 350],
            current_prices=[300, 250],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        ))

        positions.append(Position(
            strategy_type=StrategyType.SHORT_STRADDLE,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[9800, 9800],
            option_types=[OptionType.CALL, OptionType.PUT],
            quantities=[-1, -1],
            entry_prices=[400, 350],
            current_prices=[300, 250],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        ))

        positions.append(Position(
            strategy_type=StrategyType.LONG_STRANGLE,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[9500, 10100],
            option_types=[OptionType.PUT, OptionType.CALL],
            quantities=[1, 1],
            entry_prices=[300, 250],
            current_prices=[200, 150],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        ))

        positions.append(Position(
            strategy_type=StrategyType.SHORT_STRANGLE,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[9500, 10100],
            option_types=[OptionType.PUT, OptionType.CALL],
            quantities=[-1, -1],
            entry_prices=[300, 250],
            current_prices=[200, 150],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        ))

        # Complex strategies
        positions.append(Position(
            strategy_type=StrategyType.IRON_CONDOR,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[9000, 9500, 10100, 10600],
            option_types=[OptionType.PUT, OptionType.PUT, OptionType.CALL, OptionType.CALL],
            quantities=[1, -1, -1, 1],  # Long-short-short-long
            entry_prices=[150, 300, 250, 100],
            current_prices=[75, 200, 150, 50],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        ))

        positions.append(Position(
            strategy_type=StrategyType.BUTTERFLY,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[9500, 9800, 10100],
            option_types=[OptionType.CALL, OptionType.CALL, OptionType.CALL],
            quantities=[1, -2, 1],  # Long-short-short-long
            entry_prices=[450, 300, 150],
            current_prices=[350, 200, 75],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        ))

        return positions

    def test_encoder_initialization(self, encoder):
        """Test SpatialEncoder initialization."""
        assert encoder.board_rows == BOARD_ROWS
        assert encoder.board_cols == BOARD_COLS
        assert isinstance(encoder.zone_to_row, dict)
        assert len(encoder.zone_to_row) == 7  # All PositionZones
        assert isinstance(encoder.option_encoding, dict)

    def test_custom_config_initialization(self, custom_encoder):
        """Test SpatialEncoder with custom configuration."""
        assert custom_encoder.config.encoding_scale == 2.0
        assert custom_encoder.config.normalize_output is False
        assert custom_encoder.board_rows == 7
        assert custom_encoder.board_cols == 6

    def test_position_to_spatial_basic(self, encoder, sample_position):
        """Test basic position to spatial conversion."""
        spatial_tensor = encoder.position_to_spatial(sample_position)

        # Check output shape
        assert spatial_tensor.shape == (7, 6)
        assert isinstance(spatial_tensor, torch.Tensor)

        # Check for valid values (no NaN, inf)
        assert not torch.any(torch.isnan(spatial_tensor))
        assert not torch.any(torch.isinf(spatial_tensor))

        # Check that some cells have non-zero values (position has legs)
        assert torch.any(spatial_tensor != 0)

    def test_all_strategy_types_encoding(self, encoder, all_strategy_positions):
        """Test that all 16 strategy types produce valid 7x6 spatial tensors."""
        for position in all_strategy_positions:
            spatial_tensor = encoder.position_to_spatial(position)

            # Verify shape
            assert spatial_tensor.shape == (7, 6), \
                f"Strategy {position.strategy_type} produced wrong shape: {spatial_tensor.shape}"

            # Verify valid values
            assert not torch.any(torch.isnan(spatial_tensor)), \
                f"Strategy {position.strategy_type} produced NaN values"
            assert not torch.any(torch.isinf(spatial_tensor)), \
                f"Strategy {position.strategy_type} produced infinite values"

    def test_batch_encoding(self, encoder, all_strategy_positions):
        """Test batch encoding functionality."""
        batch_tensor = encoder.encode_batch(all_strategy_positions)

        # Check batch shape
        expected_shape = (len(all_strategy_positions), 7, 6)
        assert batch_tensor.shape == expected_shape

        # Check individual tensors
        for i, position in enumerate(all_strategy_positions):
            individual_tensor = encoder.position_to_spatial(position)
            batch_individual = batch_tensor[i]

            # Should be identical
            assert torch.allclose(individual_tensor, batch_individual, atol=1e-6)

    def test_validate_spatial_dimensions(self, encoder):
        """Test spatial dimension validation."""
        # Valid tensor should pass
        valid_tensor = torch.zeros((7, 6))
        encoder.validate_spatial_dimensions(valid_tensor)  # Should not raise

        # Invalid shapes should raise
        with pytest.raises(ValueError, match="Spatial tensor must have shape"):
            encoder.validate_spatial_dimensions(torch.zeros((5, 6)))

        with pytest.raises(ValueError, match="Spatial tensor must have shape"):
            encoder.validate_spatial_dimensions(torch.zeros((7, 4)))

        # NaN values should raise
        nan_tensor = torch.zeros((7, 6))
        nan_tensor[0, 0] = torch.nan
        with pytest.raises(ValueError, match="contains NaN values"):
            encoder.validate_spatial_dimensions(nan_tensor)

        # Infinite values should raise
        inf_tensor = torch.zeros((7, 6))
        inf_tensor[0, 0] = torch.inf
        with pytest.raises(ValueError, match="contains infinite values"):
            encoder.validate_spatial_dimensions(inf_tensor)

    def test_forward_method(self, encoder, sample_position):
        """Test forward method compatibility with PyTorch."""
        # Forward should work the same as position_to_spatial
        forward_result = encoder.forward(sample_position)
        direct_result = encoder.position_to_spatial(sample_position)

        assert torch.allclose(forward_result, direct_result)
        assert forward_result.shape == (7, 6)

    def test_option_type_encoding(self, encoder):
        """Test option type encoding values."""
        # Create positions with different option types
        base_date = datetime.now()

        call_position = Position(
            strategy_type=StrategyType.LONG_CALL,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[10000],
            option_types=[OptionType.CALL],
            quantities=[1],
            entry_prices=[500],
            current_prices=[300],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        )

        put_position = Position(
            strategy_type=StrategyType.LONG_PUT,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[9500],
            option_types=[OptionType.PUT],
            quantities=[1],
            entry_prices=[400],
            current_prices=[200],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        )

        call_tensor = encoder.position_to_spatial(call_position)
        put_tensor = encoder.position_to_spatial(put_position)

        # Should have different encodings for calls vs puts
        call_nonzero = call_tensor[call_tensor != 0]
        put_nonzero = put_tensor[put_tensor != 0]

        if len(call_nonzero) > 0 and len(put_nonzero) > 0:
            # Call and put encodings should have different signs or magnitudes
            call_mean = torch.mean(call_nonzero)
            put_mean = torch.mean(put_nonzero)
            assert not torch.allclose(call_mean, put_mean, atol=0.1)

    def test_quantity_influence(self, encoder):
        """Test that large quantities create spatial influence."""
        base_date = datetime.now()

        # Small quantity position
        small_qty_position = Position(
            strategy_type=StrategyType.LONG_CALL,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[10000],
            option_types=[OptionType.CALL],
            quantities=[1],
            entry_prices=[500],
            current_prices=[300],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        )

        # Large quantity position
        large_qty_position = Position(
            strategy_type=StrategyType.LONG_CALL,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[10000],
            option_types=[OptionType.CALL],
            quantities=[10],  # 10x larger
            entry_prices=[500],
            current_prices=[300],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        )

        small_tensor = encoder.position_to_spatial(small_qty_position)
        large_tensor = encoder.position_to_spatial(large_qty_position)

        # Large quantity should have more non-zero cells or higher values
        small_nonzero = torch.count_nonzero(small_tensor)
        large_nonzero = torch.count_nonzero(large_tensor)

        # Either more cells are active, or values are larger
        assert (large_nonzero >= small_nonzero or
                torch.max(torch.abs(large_tensor)) >= torch.max(torch.abs(small_tensor)))

    def test_normalization_toggle(self):
        """Test normalization configuration option."""
        # Encoder with normalization
        normalized_encoder = SpatialEncoder(SpatialConfig(normalize_output=True))

        # Encoder without normalization
        unnormalized_encoder = SpatialEncoder(SpatialConfig(normalize_output=False))

        base_date = datetime.now()
        position = Position(
            strategy_type=StrategyType.LONG_CALL,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[10000],
            option_types=[OptionType.CALL],
            quantities=[10],
            entry_prices=[500],
            current_prices=[300],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        )

        norm_tensor = normalized_encoder.position_to_spatial(position)
        unnorm_tensor = unnormalized_encoder.position_to_spatial(position)

        # Normalized tensor should be bounded
        assert torch.max(torch.abs(norm_tensor)) <= 1.1  # Allow small epsilon

        # Unnormalized tensor may have larger values
        # They should be different (unless position is very simple)
        if torch.any(unnorm_tensor != 0):
            assert not torch.allclose(norm_tensor, unnorm_tensor, atol=0.1)

    def test_spatial_info_decoding(self, encoder, sample_position):
        """Test spatial tensor information extraction."""
        spatial_tensor = encoder.position_to_spatial(sample_position)
        info = encoder.decode_spatial_info(spatial_tensor)

        # Check info structure
        assert 'shape' in info
        assert 'non_zero_cells' in info
        assert 'value_range' in info
        assert 'active_zones' in info
        assert 'active_columns' in info

        # Check values
        assert info['shape'] == (7, 6)
        assert info['non_zero_cells'] >= 0
        assert isinstance(info['value_range'], dict)
        assert isinstance(info['active_zones'], list)
        assert isinstance(info['active_columns'], list)

    def test_visualization(self, encoder, sample_position):
        """Test spatial tensor visualization."""
        spatial_tensor = encoder.position_to_spatial(sample_position)
        viz = encoder.visualize_spatial_tensor(spatial_tensor)

        # Should be a string with multiple lines
        assert isinstance(viz, str)
        assert len(viz.split('\n')) >= 7  # At least 7 rows
        assert 'MAX_PROFIT' in viz
        assert 'DEEP_LOSS' in viz

    def test_edge_cases(self, encoder):
        """Test edge cases and boundary conditions."""
        base_date = datetime.now()

        # Position with zero prices
        zero_price_position = Position(
            strategy_type=StrategyType.LONG_CALL,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[10000],
            option_types=[OptionType.CALL],
            quantities=[1],
            entry_prices=[0],  # Zero entry price
            current_prices=[0],  # Zero current price
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        )

        # Should handle gracefully
        tensor = encoder.position_to_spatial(zero_price_position)
        assert tensor.shape == (7, 6)
        assert not torch.any(torch.isnan(tensor))

        # Position with very large strikes
        large_strike_position = Position(
            strategy_type=StrategyType.LONG_CALL,
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[50000],  # Very high strike
            option_types=[OptionType.CALL],
            quantities=[1],
            entry_prices=[100],
            current_prices=[50],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        )

        # Should handle gracefully
        tensor = encoder.position_to_spatial(large_strike_position)
        assert tensor.shape == (7, 6)
        assert not torch.any(torch.isnan(tensor))

    def test_max_legs_constraint(self, encoder):
        """Test that encoder handles more than 6 legs gracefully."""
        base_date = datetime.now()

        # Create position with 8 legs (more than max 6 columns)
        many_legs_position = Position(
            strategy_type=StrategyType.IRON_CONDOR,  # Use iron condor as base
            entry_date=base_date - timedelta(days=10),
            expiration_date=base_date + timedelta(days=20),
            strikes=[9000, 9200, 9400, 9600, 9800, 10000, 10200, 10400],  # 8 strikes
            option_types=[OptionType.PUT] * 4 + [OptionType.CALL] * 4,
            quantities=[1, -1, 1, -1, -1, 1, -1, 1],
            entry_prices=[100, 200, 150, 300, 250, 150, 200, 100],
            current_prices=[50, 100, 75, 150, 125, 75, 100, 50],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        )

        # Should handle gracefully and only use first 6 legs
        tensor = encoder.position_to_spatial(many_legs_position)
        assert tensor.shape == (7, 6)
        assert not torch.any(torch.isnan(tensor))

        # Should have non-zero values in maximum 6 columns
        active_cols = torch.any(tensor != 0, dim=0)
        assert torch.sum(active_cols) <= 6

    def test_repr_method(self, encoder):
        """Test string representation of encoder."""
        repr_str = repr(encoder)
        assert isinstance(repr_str, str)
        assert 'SpatialEncoder' in repr_str
        assert '7x6' in repr_str
        assert str(len(StrategyType)) in repr_str  # Should mention number of strategies

    @pytest.mark.parametrize("strategy_type", list(StrategyType))
    def test_individual_strategy_types(self, encoder, strategy_type):
        """Test each strategy type individually."""
        base_date = datetime.now()

        # Create minimal position for each strategy type
        if strategy_type in [StrategyType.LONG_CALL, StrategyType.SHORT_CALL]:
            position = Position(
                strategy_type=strategy_type,
                entry_date=base_date - timedelta(days=10),
                expiration_date=base_date + timedelta(days=20),
                strikes=[10000],
                option_types=[OptionType.CALL],
                quantities=[1 if 'LONG' in strategy_type.name else -1],
                entry_prices=[500],
                current_prices=[300],
                underlying_price_at_entry=9800,
                current_underlying_price=9900
            )
        elif strategy_type in [StrategyType.LONG_PUT, StrategyType.SHORT_PUT]:
            position = Position(
                strategy_type=strategy_type,
                entry_date=base_date - timedelta(days=10),
                expiration_date=base_date + timedelta(days=20),
                strikes=[9500],
                option_types=[OptionType.PUT],
                quantities=[1 if 'LONG' in strategy_type.name else -1],
                entry_prices=[400],
                current_prices=[200],
                underlying_price_at_entry=9800,
                current_underlying_price=9900
            )
        else:
            # Use a generic 2-leg position for more complex strategies
            position = Position(
                strategy_type=strategy_type,
                entry_date=base_date - timedelta(days=10),
                expiration_date=base_date + timedelta(days=20),
                strikes=[9500, 10000],
                option_types=[OptionType.CALL, OptionType.CALL],
                quantities=[1, -1],
                entry_prices=[600, 400],
                current_prices=[400, 200],
                underlying_price_at_entry=9800,
                current_underlying_price=9900
            )

        # Should produce valid tensor
        tensor = encoder.position_to_spatial(position)
        assert tensor.shape == (7, 6)
        assert not torch.any(torch.isnan(tensor))
        assert not torch.any(torch.isinf(tensor))
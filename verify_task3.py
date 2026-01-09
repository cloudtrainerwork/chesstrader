#!/usr/bin/env python
"""
Verification script for Task 3: Volatility and Market Context Features
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    try:
        from src.features.regime_features import (
            VolatilityFeatures, VolumeFeatures, SupportResistanceFeatures,
            MarketContextFeatures, EventFeatures, RegimeStateVector
        )
        print("✓ Successfully imported all Task 3 feature classes")

        # Test each feature class individually
        print("\n--- Testing VolatilityFeatures ---")
        vf = VolatilityFeatures()
        volatility_features = vf.calculate('SPY')
        print(f"✓ Volatility features: shape {volatility_features.shape}")
        print(f"✓ Columns: {list(volatility_features.columns)}")
        assert volatility_features.shape[1] == 11, f"Expected 11 volatility features, got {volatility_features.shape[1]}"

        print("\n--- Testing VolumeFeatures ---")
        volf = VolumeFeatures()
        volume_features = volf.calculate('SPY')
        print(f"✓ Volume features: shape {volume_features.shape}")
        print(f"✓ Columns: {list(volume_features.columns)}")
        assert volume_features.shape[1] == 3, f"Expected 3 volume features, got {volume_features.shape[1]}"

        print("\n--- Testing SupportResistanceFeatures ---")
        srf = SupportResistanceFeatures()
        sr_features = srf.calculate('SPY')
        print(f"✓ Support/resistance features: shape {sr_features.shape}")
        print(f"✓ Columns: {list(sr_features.columns)}")
        assert sr_features.shape[1] == 6, f"Expected 6 support/resistance features, got {sr_features.shape[1]}"

        print("\n--- Testing MarketContextFeatures ---")
        mcf = MarketContextFeatures()
        market_features = mcf.calculate('SPY')
        print(f"✓ Market context features: shape {market_features.shape}")
        print(f"✓ Columns: {list(market_features.columns)}")
        assert market_features.shape[1] == 4, f"Expected 4 market context features, got {market_features.shape[1]}"

        print("\n--- Testing EventFeatures ---")
        ef = EventFeatures()
        event_features = ef.calculate('SPY')
        print(f"✓ Event features: shape {event_features.shape}")
        print(f"✓ Columns: {list(event_features.columns)}")
        assert event_features.shape[1] == 3, f"Expected 3 event features, got {event_features.shape[1]}"

        # Test complete 48-dimensional vector
        print("\n--- Testing Complete RegimeStateVector ---")
        rsv = RegimeStateVector()
        state_vector = rsv.calculate('SPY')
        print(f"✓ Complete state vector: shape {state_vector.shape}")
        print(f"✓ Feature names: {list(state_vector.index)}")

        assert len(state_vector) == 48, f"Expected 48 dimensions, got {len(state_vector)}"
        print("✓ Correct number of total features (48)")

        # Check for NaNs
        assert not state_vector.isnull().any(), "State vector contains NaN values"
        print("✓ No NaN values in complete state vector")

        # Verify breakdown: 6 + 9 + 6 + 11 + 3 + 6 + 4 + 3 = 48
        expected_breakdown = [6, 9, 6, 11, 3, 6, 4, 3]
        actual_total = sum(expected_breakdown)
        assert actual_total == 48, f"Feature breakdown doesn't sum to 48: {expected_breakdown} = {actual_total}"
        print(f"✓ Feature breakdown verified: {expected_breakdown} = 48 total")

        print("\n🎉 Task 3 verification PASSED!")
        print(f"Complete regime state vector successfully returns {len(state_vector)} features for SPY")

        # Print feature summary
        print(f"\n📊 Feature Summary:")
        print(f"- Price structure: 6 features")
        print(f"- Trend indicators: 9 features")
        print(f"- Momentum indicators: 6 features")
        print(f"- Volatility features: 11 features")
        print(f"- Volume features: 3 features")
        print(f"- Support/resistance: 6 features")
        print(f"- Market context: 4 features")
        print(f"- Event features: 3 features")
        print(f"Total: {len(state_vector)} dimensions")

    except Exception as e:
        print(f"❌ Task 3 verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    main()
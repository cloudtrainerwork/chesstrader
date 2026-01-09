#!/usr/bin/env python
"""
Verification script for Task 1: Price Structure Features
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    try:
        from src.features.regime_features import PriceStructureFeatures
        print("✓ Successfully imported PriceStructureFeatures")

        pf = PriceStructureFeatures()
        print("✓ Successfully created PriceStructureFeatures instance")

        # Test with SPY
        features = pf.calculate('SPY')
        print(f"✓ Features calculated for SPY: shape {features.shape}")
        print(f"✓ Columns: {list(features.columns)}")

        # Verify shape
        assert features.shape[1] == 6, f"Expected 6 features, got {features.shape[1]}"
        print("✓ Correct number of features (6)")

        # Verify column names
        expected_columns = [
            'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200',
            'distance_from_52w_high', 'distance_from_52w_low', 'gap_percentage'
        ]
        assert list(features.columns) == expected_columns, "Column names mismatch"
        print("✓ Correct column names")

        # Check for NaNs
        assert not features.isnull().any().any(), "Features contain NaN values"
        print("✓ No NaN values")

        print("\n🎉 Task 1 verification PASSED!")
        print(f"Price structure features successfully return {features.shape[1]} normalized dimensions for SPY")

    except Exception as e:
        print(f"❌ Task 1 verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    main()
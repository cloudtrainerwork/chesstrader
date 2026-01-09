#!/usr/bin/env python
"""
Verification script for Task 2: Trend and Momentum Indicators
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    try:
        from src.features.regime_features import TrendIndicators, MomentumIndicators
        print("✓ Successfully imported TrendIndicators and MomentumIndicators")

        # Test Trend Indicators
        tf = TrendIndicators()
        print("✓ Successfully created TrendIndicators instance")

        trend_features = tf.calculate('SPY')
        print(f"✓ Trend features calculated: shape {trend_features.shape}")
        print(f"✓ Trend columns: {list(trend_features.columns)}")

        # Verify trend features shape
        assert trend_features.shape[1] == 9, f"Expected 9 trend features, got {trend_features.shape[1]}"
        print("✓ Correct number of trend features (9)")

        # Test Momentum Indicators
        mf = MomentumIndicators()
        print("✓ Successfully created MomentumIndicators instance")

        momentum_features = mf.calculate('SPY')
        print(f"✓ Momentum features calculated: shape {momentum_features.shape}")
        print(f"✓ Momentum columns: {list(momentum_features.columns)}")

        # Verify momentum features shape
        assert momentum_features.shape[1] == 6, f"Expected 6 momentum features, got {momentum_features.shape[1]}"
        print("✓ Correct number of momentum features (6)")

        # Check for NaNs
        assert not trend_features.isnull().any().any(), "Trend features contain NaN values"
        assert not momentum_features.isnull().any().any(), "Momentum features contain NaN values"
        print("✓ No NaN values in trend or momentum features")

        # Total should be 15 (9 + 6)
        total_features = trend_features.shape[1] + momentum_features.shape[1]
        assert total_features == 15, f"Expected 15 total features, got {total_features}"
        print(f"✓ Total trend + momentum features: {total_features}")

        print("\n🎉 Task 2 verification PASSED!")
        print(f"Trend and momentum indicators successfully return {total_features} features for SPY")

    except Exception as e:
        print(f"❌ Task 2 verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    main()
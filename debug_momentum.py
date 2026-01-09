#!/usr/bin/env python
"""
Debug script for momentum indicators NaN issue
"""
import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    from src.features.regime_features import MomentumIndicators

    mf = MomentumIndicators()
    df = mf.get_data('SPY', days=300)
    print(f"Data shape: {df.shape}")
    print(f"Data columns: {df.columns.tolist()}")
    print(f"Data has NaNs: {df.isnull().any().any()}")

    features = pd.DataFrame(index=df.index)

    # Test RSI calculation
    print("\n--- Testing RSI ---")
    rsi = mf.calculate_rsi(df['Close'])
    print(f"RSI has NaNs: {rsi.isnull().any()}")
    print(f"RSI NaN count: {rsi.isnull().sum()}")
    features['rsi'] = (rsi / 50) - 1

    # Test Stochastic calculation
    print("\n--- Testing Stochastic ---")
    stoch_k, stoch_d = mf.calculate_stochastic(df['High'], df['Low'], df['Close'])
    print(f"Stoch K has NaNs: {stoch_k.isnull().any()}")
    print(f"Stoch D has NaNs: {stoch_d.isnull().any()}")
    features['stoch_k'] = (stoch_k / 50) - 1
    features['stoch_d'] = (stoch_d / 50) - 1

    # Test ROC calculation
    print("\n--- Testing ROC ---")
    roc_5 = df['Close'].pct_change(periods=5)
    roc_10 = df['Close'].pct_change(periods=10)
    roc_20 = df['Close'].pct_change(periods=20)
    print(f"ROC 5 has NaNs: {roc_5.isnull().any()}")
    print(f"ROC 10 has NaNs: {roc_10.isnull().any()}")
    print(f"ROC 20 has NaNs: {roc_20.isnull().any()}")
    features['roc_5'] = roc_5
    features['roc_10'] = roc_10
    features['roc_20'] = roc_20

    print(f"\n--- Features before missing data handling ---")
    print(f"Features shape: {features.shape}")
    for col in features.columns:
        nan_count = features[col].isnull().sum()
        print(f"{col}: {nan_count} NaNs")

    # Handle missing data
    features = mf.handle_missing_data(features)
    print(f"\n--- Features after missing data handling ---")
    for col in features.columns:
        nan_count = features[col].isnull().sum()
        print(f"{col}: {nan_count} NaNs")

    # Try standardization
    print(f"\n--- Testing standardization ---")
    for col in ['roc_5', 'roc_10', 'roc_20']:
        print(f"Standardizing {col}...")
        try:
            std_values = mf.standardize(features[col], method='robust')
            print(f"{col} standardized, NaN count: {std_values.isnull().sum()}")
        except Exception as e:
            print(f"Error standardizing {col}: {e}")

if __name__ == "__main__":
    main()
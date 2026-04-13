#!/usr/bin/env python3
"""
Simple test of options data access
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import yfinance as yf

    print("Testing options chain access...")
    stock = yf.Ticker("AAPL")

    # Get current price
    hist = stock.history(period="1d")
    current_price = float(hist['Close'].iloc[-1])
    print(f"Current AAPL price: ${current_price:.2f}")

    # Get expirations
    expirations = stock.options
    print(f"Found {len(expirations)} expirations")

    if expirations:
        exp = expirations[0]
        print(f"Using expiration: {exp}")

        # Get options chain
        option_chain = stock.option_chain(exp)
        calls = option_chain.calls

        print(f"Calls dataframe shape: {calls.shape}")
        print(f"Calls columns: {list(calls.columns)}")

        # Show first few rows with relevant columns
        if not calls.empty:
            relevant_cols = ['strike', 'bid', 'ask', 'lastPrice', 'volume']
            available_cols = [col for col in relevant_cols if col in calls.columns]
            print(f"Available relevant columns: {available_cols}")

            print("First 3 calls:")
            print(calls[available_cols].head(3))

            # Test accessing a specific strike
            target_strike = calls['strike'].iloc[2]  # 3rd option
            matching_call = calls[calls['strike'] == target_strike].iloc[0]
            print(f"\nTesting strike ${target_strike}:")
            print(f"Type: {type(matching_call)}")
            print(f"Ask price: {matching_call['ask'] if 'ask' in matching_call else 'N/A'}")
            print(f"Bid price: {matching_call['bid'] if 'bid' in matching_call else 'N/A'}")

    else:
        print("No expirations found!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
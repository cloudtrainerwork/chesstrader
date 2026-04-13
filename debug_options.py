#!/usr/bin/env python3
"""
Debug script to check options chain data
"""

import yfinance as yf
import pandas as pd

def debug_options_chain(symbol="AAPL"):
    print(f"=== Debugging Options Chain for {symbol} ===")

    try:
        stock = yf.Ticker(symbol)

        # Get expirations
        expirations = stock.options
        print(f"Available expirations: {expirations[:3]}")  # First 3

        if not expirations:
            print("❌ No expirations found!")
            return

        # Get option chain for first expiration
        exp = expirations[0]
        print(f"Using expiration: {exp}")

        option_chain = stock.option_chain(exp)

        print(f"\n=== CALLS ===")
        calls = option_chain.calls
        print(f"Calls shape: {calls.shape}")
        print(f"Calls columns: {list(calls.columns)}")
        print(f"Sample calls data:")
        print(calls[['strike', 'bid', 'ask', 'lastPrice', 'volume']].head())

        print(f"\n=== PUTS ===")
        puts = option_chain.puts
        print(f"Puts shape: {puts.shape}")
        print(f"Puts columns: {list(puts.columns)}")
        print(f"Sample puts data:")
        print(puts[['strike', 'bid', 'ask', 'lastPrice', 'volume']].head())

        # Check current stock price
        hist = stock.history(period="1d")
        current_price = hist['Close'].iloc[-1]
        print(f"\nCurrent stock price: ${current_price:.2f}")

        # Find ATM options
        atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
        atm_put = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]

        print(f"\n=== ATM CALL ===")
        print(atm_call[['strike', 'bid', 'ask', 'lastPrice']])

        print(f"\n=== ATM PUT ===")
        print(atm_put[['strike', 'bid', 'ask', 'lastPrice']])

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_options_chain("AAPL")
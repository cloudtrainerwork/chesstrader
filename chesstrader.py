#!/usr/bin/env python3
"""
ChessTrader - AI-Powered Options Trading System

Convenience entry point for running ChessTrader without installation.
For production use, install with: pip install -e .

Usage:
    python chesstrader.py --help
    python chesstrader.py recommend AAPL
    python chesstrader.py backtest --symbol SPY
"""

import sys
import os

# Add src directory to Python path for development usage
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    # First try package import, then fallback to development
    try:
        from src.cli.main import app
    except ImportError:
        # Try adding src to path and importing directly
        try:
            import sys
            from pathlib import Path
            src_path = Path(__file__).parent / 'src'
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            from cli.main import app
        except ImportError:
            from cli.main import app
except ImportError as e:
    print(f"Error importing ChessTrader CLI: {e}")
    print("\nPlease ensure all dependencies are installed:")
    print("  pip install -r requirements.txt")
    print("\nOr install ChessTrader as a package:")
    print("  pip install -e .")
    sys.exit(1)

if __name__ == "__main__":
    # Run the Typer CLI application
    app()
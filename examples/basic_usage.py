#!/usr/bin/env python3
"""
ChessTrader Basic Usage Example

Demonstrates fundamental usage of the OptionsAI system for strategy recommendations.

This example covers:
- System initialization
- Getting AI-powered strategy recommendations
- Exploring strategy details
- Configuration management
- Error handling patterns

Run with: python examples/basic_usage.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for development usage
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from main import OptionsAI
    from config import Config
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory.")
    print("Or install ChessTrader: pip install -e .")
    sys.exit(1)


async def basic_recommendations_example():
    """Basic example of getting strategy recommendations."""
    print("🎯 Basic Strategy Recommendations Example")
    print("=" * 50)

    # Initialize OptionsAI with default configuration
    ai = OptionsAI()

    # Example symbols to analyze
    symbols = ['AAPL', 'SPY', 'MSFT']

    for symbol in symbols:
        try:
            print(f"\n📊 Analyzing {symbol}...")

            # Get recommendations for the symbol
            recommendations = await ai.get_recommendations(symbol)

            if recommendations:
                print(f"Found {len(recommendations)} recommendations for {symbol}:")

                for i, rec in enumerate(recommendations, 1):
                    confidence_emoji = "🟢" if rec['confidence'] >= 0.8 else "🟡" if rec['confidence'] >= 0.6 else "🔴"
                    strategy_name = rec['strategy'].replace('_', ' ').title()

                    print(f"  {i}. {confidence_emoji} {strategy_name}")
                    print(f"     Confidence: {rec['confidence']:.1%}")
                    print(f"     Score: {rec['score']}")
                    print(f"     Market Outlook: {rec['market_outlook']}")

            else:
                print(f"❌ No recommendations found for {symbol}")

        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")

    print("\n✅ Basic recommendations example completed!")


async def strategy_details_example():
    """Example of exploring strategy details."""
    print("\n🔍 Strategy Details Exploration Example")
    print("=" * 50)

    ai = OptionsAI()

    # Get recommendations for a symbol
    symbol = 'AAPL'
    print(f"Getting recommendations for {symbol}...")

    try:
        recommendations = await ai.get_recommendations(symbol)

        if recommendations:
            # Explore details of each recommended strategy
            print(f"\n📋 Detailed Analysis for {symbol}:")

            for i, rec in enumerate(recommendations, 1):
                strategy_name = rec['strategy']
                print(f"\n--- Strategy {i}: {strategy_name.replace('_', ' ').title()} ---")

                try:
                    # Get detailed strategy information
                    details = ai.get_strategy_details(strategy_name)

                    print(f"Description: {details.get('description', 'No description available')}")
                    print(f"Risk Profile: {details.get('risk_profile', 'Unknown').title()}")
                    print(f"Market Outlook: {details.get('market_outlook', 'Unknown')}")
                    print(f"Typical Duration: {details.get('typical_duration', 'Unknown')}")

                    # Display recommendation confidence and score
                    print(f"AI Confidence: {rec['confidence']:.1%}")
                    print(f"Strategy Score: {rec['score']}")

                except Exception as e:
                    print(f"Could not get details for {strategy_name}: {e}")

        else:
            print(f"No recommendations available for {symbol}")

    except Exception as e:
        print(f"Error getting recommendations: {e}")

    print("\n✅ Strategy details example completed!")


async def configuration_example():
    """Example of configuration management."""
    print("\n⚙️ Configuration Management Example")
    print("=" * 50)

    # Create custom configuration
    custom_config = Config()
    custom_config.recommendation.confidence_threshold = 0.7
    custom_config.recommendation.max_recommendations = 5

    # Initialize OptionsAI with custom configuration
    ai = OptionsAI(config=custom_config)

    print(f"Configuration settings:")
    print(f"  Confidence threshold: {ai.config.recommendation.confidence_threshold:.1%}")
    print(f"  Max recommendations: {ai.config.recommendation.max_recommendations}")
    print(f"  Initial capital: ${ai.config.backtesting.initial_capital:,.0f}")
    print(f"  Commission: ${ai.config.backtesting.commission}")

    # Demonstrate dynamic configuration updates
    print(f"\n🔧 Updating configuration dynamically...")

    ai.update_config(
        recommendation__confidence_threshold=0.8,
        recommendation__max_recommendations=2
    )

    print(f"Updated settings:")
    print(f"  Confidence threshold: {ai.config.recommendation.confidence_threshold:.1%}")
    print(f"  Max recommendations: {ai.config.recommendation.max_recommendations}")

    # Test with updated configuration
    symbol = 'SPY'
    print(f"\n📊 Testing with high confidence threshold on {symbol}...")

    try:
        recommendations = await ai.get_recommendations(symbol)

        if recommendations:
            print(f"High-confidence recommendations for {symbol}:")
            for rec in recommendations:
                print(f"  • {rec['strategy'].replace('_', ' ').title()}: {rec['confidence']:.1%}")
        else:
            print(f"No high-confidence recommendations found for {symbol}")
            print("Consider lowering the confidence threshold.")

    except Exception as e:
        print(f"Error: {e}")

    print("\n✅ Configuration example completed!")


async def error_handling_example():
    """Example of proper error handling patterns."""
    print("\n🛡️ Error Handling Patterns Example")
    print("=" * 50)

    ai = OptionsAI()

    # Test cases that may produce errors
    test_cases = [
        ('INVALID_SYMBOL', 'Invalid symbol'),
        ('', 'Empty symbol'),
        ('AAPL', 'Valid symbol (should work)')
    ]

    for symbol, description in test_cases:
        print(f"\n🧪 Testing: {description}")
        print(f"Symbol: '{symbol}'")

        try:
            if symbol:
                recommendations = await ai.get_recommendations(symbol)

                if recommendations:
                    top_rec = recommendations[0]
                    print(f"✅ Success: {top_rec['strategy']} ({top_rec['confidence']:.1%})")
                else:
                    print("⚠️ No recommendations found")

            else:
                print("❌ Empty symbol provided")

        except ValueError as e:
            print(f"❌ Validation error: {e}")
        except ConnectionError as e:
            print(f"❌ Network error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

    # Test strategy details error handling
    print(f"\n🧪 Testing strategy details error handling...")

    test_strategies = ['iron_condor', 'invalid_strategy', '']

    for strategy in test_strategies:
        try:
            if strategy:
                details = ai.get_strategy_details(strategy)
                print(f"✅ {strategy}: {details.get('description', 'No description')[:50]}...")
            else:
                print("❌ Empty strategy name")

        except KeyError as e:
            print(f"❌ Strategy not found: {strategy}")
        except Exception as e:
            print(f"❌ Error getting details for '{strategy}': {e}")

    print("\n✅ Error handling example completed!")


async def advanced_usage_example():
    """Example of advanced usage patterns."""
    print("\n🚀 Advanced Usage Patterns Example")
    print("=" * 50)

    ai = OptionsAI()

    # Multi-symbol batch analysis
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ']
    results = []

    print("📊 Batch Analysis of Multiple Symbols")

    for symbol in symbols:
        try:
            recommendations = await ai.get_recommendations(symbol)

            if recommendations:
                top_rec = recommendations[0]
                results.append({
                    'symbol': symbol,
                    'strategy': top_rec['strategy'],
                    'confidence': top_rec['confidence'],
                    'score': top_rec['score']
                })

                print(f"  {symbol}: {top_rec['strategy'].replace('_', ' ').title()} "
                      f"({top_rec['confidence']:.1%} confidence)")

        except Exception as e:
            print(f"  {symbol}: Error - {e}")

    # Analysis of results
    if results:
        print(f"\n📈 Batch Analysis Summary:")
        print(f"  Analyzed symbols: {len(results)}")

        high_confidence = [r for r in results if r['confidence'] >= 0.7]
        print(f"  High confidence (≥70%): {len(high_confidence)}")

        if high_confidence:
            print(f"  Top recommendations:")
            sorted_results = sorted(high_confidence, key=lambda x: x['confidence'], reverse=True)
            for result in sorted_results[:3]:
                print(f"    • {result['symbol']}: {result['strategy'].replace('_', ' ').title()} "
                      f"({result['confidence']:.1%})")

    # Configuration persistence example
    print(f"\n💾 Configuration Persistence Example")

    # Update configuration
    ai.update_config(
        recommendation__confidence_threshold=0.75,
        backtesting__initial_capital=200000,
        backtesting__commission=0.50
    )

    # Save configuration to file
    config_file = "examples/temp_config.json"
    try:
        ai.save_config(config_file)
        print(f"✅ Configuration saved to {config_file}")

        # Load configuration from file
        ai2 = OptionsAI(config_path=config_file)
        print(f"✅ Configuration loaded from file")
        print(f"  Confidence threshold: {ai2.config.recommendation.confidence_threshold:.1%}")
        print(f"  Initial capital: ${ai2.config.backtesting.initial_capital:,.0f}")

        # Clean up
        import os
        if os.path.exists(config_file):
            os.remove(config_file)
            print(f"🗑️ Cleaned up {config_file}")

    except Exception as e:
        print(f"❌ Configuration persistence error: {e}")

    print("\n✅ Advanced usage example completed!")


async def main():
    """Run all examples."""
    print("🏛️ ChessTrader Options AI - Basic Usage Examples")
    print("=" * 60)
    print()
    print("This example demonstrates the fundamental capabilities of ChessTrader:")
    print("• AI-powered options strategy recommendations")
    print("• Strategy analysis and detailed information")
    print("• Configuration management")
    print("• Error handling patterns")
    print("• Advanced usage techniques")
    print()
    print("Note: This example uses mock data when real market data is unavailable.")
    print()

    try:
        # Run all example functions
        await basic_recommendations_example()
        await strategy_details_example()
        await configuration_example()
        await error_handling_example()
        await advanced_usage_example()

        print("\n🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("• Try the backtest example: python examples/backtest_example.py")
        print("• Explore the CLI: chesstrader --help")
        print("• Read the documentation: docs/api_reference.md")

    except KeyboardInterrupt:
        print("\n\n⚠️ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error in examples: {e}")
        print("This may be due to missing dependencies or configuration issues.")
        print("Please check the installation and try again.")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
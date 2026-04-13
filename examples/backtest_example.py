#!/usr/bin/env python3
"""
ChessTrader Backtesting Example

Demonstrates comprehensive backtesting workflow using the OptionsAI system.

This example covers:
- Basic backtesting setup and execution
- Performance analysis and interpretation
- Strategy comparison workflows
- Configuration for different trading scenarios
- Advanced backtesting techniques

Run with: python examples/backtest_example.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

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


async def basic_backtest_example():
    """Basic backtesting example with single strategy."""
    print("🔬 Basic Backtesting Example")
    print("=" * 40)

    ai = OptionsAI()

    # Configure basic backtest parameters
    config = {
        'symbol': 'SPY',  # S&P 500 ETF
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'initial_capital': 100000,
        'strategy': 'iron_condor'
    }

    print(f"Backtesting Configuration:")
    print(f"  Symbol: {config['symbol']}")
    print(f"  Strategy: {config['strategy'].replace('_', ' ').title()}")
    print(f"  Period: {config['start_date']} to {config['end_date']}")
    print(f"  Initial Capital: ${config['initial_capital']:,.0f}")
    print()

    try:
        print("⏳ Running backtest (this may take a moment)...")
        results = await ai.run_backtest(config)

        if results:
            print("✅ Backtest completed successfully!\n")

            # Display key performance metrics
            print("📊 Performance Summary:")
            print(f"  Total Return: {results.get('total_return', 0.0):.2%}")
            print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0.0):.2f}")
            print(f"  Maximum Drawdown: {results.get('max_drawdown', 0.0):.2%}")
            print(f"  Win Rate: {results.get('win_rate', 0.0):.1%}")
            print(f"  Total Trades: {results.get('total_trades', 0)}")
            print(f"  Average Trade Return: {results.get('avg_trade_return', 0.0):.2%}")

            # Risk metrics
            print(f"\n⚠️ Risk Analysis:")
            print(f"  Volatility: {results.get('volatility', 0.0):.2%}")
            print(f"  Sortino Ratio: {results.get('sortino_ratio', 0.0):.2f}")
            print(f"  Value at Risk (95%): {results.get('var_95', 0.0):.2%}")

            # Performance assessment
            total_return = results.get('total_return', 0.0)
            sharpe_ratio = results.get('sharpe_ratio', 0.0)
            max_drawdown = results.get('max_drawdown', 0.0)

            if total_return > 0.15 and sharpe_ratio > 1.0 and max_drawdown > -0.15:
                assessment = "🌟 Excellent"
            elif total_return > 0.05 and sharpe_ratio > 0.5:
                assessment = "✅ Good"
            elif total_return > 0:
                assessment = "⚡ Acceptable"
            else:
                assessment = "⚠️ Needs Improvement"

            print(f"\n🎯 Overall Assessment: {assessment}")

        else:
            print("❌ No backtest results generated")
            print("This could be due to insufficient historical data or configuration issues.")

    except Exception as e:
        print(f"❌ Backtest failed: {e}")

    print("\n✅ Basic backtest example completed!")


async def strategy_comparison_example():
    """Example comparing multiple strategies on the same symbol."""
    print("\n🏁 Strategy Comparison Example")
    print("=" * 40)

    ai = OptionsAI()

    # Strategies to compare
    strategies = [
        'iron_condor',
        'bull_put_spread',
        'calendar_spread',
        'covered_call'
    ]

    # Common backtest configuration
    base_config = {
        'symbol': 'AAPL',
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'initial_capital': 100000
    }

    print(f"Comparing strategies on {base_config['symbol']}:")
    print(f"Period: {base_config['start_date']} to {base_config['end_date']}")
    print()

    results_comparison = []

    for strategy in strategies:
        print(f"⏳ Testing {strategy.replace('_', ' ').title()}...")

        try:
            # Create config for this strategy
            config = base_config.copy()
            config['strategy'] = strategy

            # Run backtest
            results = await ai.run_backtest(config)

            if results:
                strategy_results = {
                    'strategy': strategy,
                    'total_return': results.get('total_return', 0.0),
                    'sharpe_ratio': results.get('sharpe_ratio', 0.0),
                    'max_drawdown': results.get('max_drawdown', 0.0),
                    'win_rate': results.get('win_rate', 0.0),
                    'total_trades': results.get('total_trades', 0)
                }

                results_comparison.append(strategy_results)
                print(f"  ✅ Return: {strategy_results['total_return']:.2%}, "
                      f"Sharpe: {strategy_results['sharpe_ratio']:.2f}")

            else:
                print(f"  ❌ No results for {strategy}")

        except Exception as e:
            print(f"  ❌ Error testing {strategy}: {e}")

    # Analyze comparison results
    if results_comparison:
        print(f"\n📊 Strategy Comparison Results:")
        print("=" * 80)
        print(f"{'Strategy':<20} {'Return':<10} {'Sharpe':<8} {'Drawdown':<10} {'Win Rate':<10} {'Trades':<8}")
        print("-" * 80)

        # Sort by Sharpe ratio (risk-adjusted returns)
        sorted_results = sorted(results_comparison, key=lambda x: x['sharpe_ratio'], reverse=True)

        for result in sorted_results:
            strategy_name = result['strategy'].replace('_', ' ').title()
            print(f"{strategy_name:<20} "
                  f"{result['total_return']:>8.2%} "
                  f"{result['sharpe_ratio']:>7.2f} "
                  f"{result['max_drawdown']:>9.2%} "
                  f"{result['win_rate']:>8.1%} "
                  f"{result['total_trades']:>7}")

        # Winner analysis
        best_strategy = sorted_results[0]
        print(f"\n🏆 Best Performer (by Sharpe ratio):")
        print(f"  {best_strategy['strategy'].replace('_', ' ').title()}")
        print(f"  Total Return: {best_strategy['total_return']:.2%}")
        print(f"  Sharpe Ratio: {best_strategy['sharpe_ratio']:.2f}")

    print("\n✅ Strategy comparison example completed!")


async def custom_configuration_example():
    """Example with custom backtesting configuration."""
    print("\n⚙️ Custom Configuration Example")
    print("=" * 40)

    # Create custom configuration for conservative trading
    conservative_config = Config()
    conservative_config.backtesting.initial_capital = 50000  # Smaller account
    conservative_config.backtesting.commission = 1.0        # Higher commission
    conservative_config.backtesting.max_position_size = 3   # Smaller positions
    conservative_config.recommendation.confidence_threshold = 0.8  # High confidence only

    ai = OptionsAI(config=conservative_config)

    print("Conservative Trading Configuration:")
    print(f"  Initial Capital: ${ai.config.backtesting.initial_capital:,.0f}")
    print(f"  Commission: ${ai.config.backtesting.commission}")
    print(f"  Max Position Size: {ai.config.backtesting.max_position_size} contracts")
    print(f"  Confidence Threshold: {ai.config.recommendation.confidence_threshold:.1%}")
    print()

    # Test with conservative settings
    backtest_config = {
        'symbol': 'SPY',
        'start_date': '2023-06-01',
        'end_date': '2023-12-31',
        'strategy': 'iron_condor'
    }

    try:
        print("⏳ Running conservative backtest...")
        results = await ai.run_backtest(backtest_config)

        if results:
            print("✅ Conservative backtest results:")
            print(f"  Total Return: {results.get('total_return', 0.0):.2%}")
            print(f"  Max Drawdown: {results.get('max_drawdown', 0.0):.2%}")
            print(f"  Win Rate: {results.get('win_rate', 0.0):.1%}")
            print(f"  Total Trades: {results.get('total_trades', 0)}")

            # Analyze conservative performance
            if results.get('max_drawdown', 0.0) > -0.1:
                print("  🛡️ Low drawdown achieved - conservative approach working!")
            else:
                print("  ⚠️ Consider even more conservative position sizing")

    except Exception as e:
        print(f"❌ Conservative backtest failed: {e}")

    # Now try aggressive configuration
    print(f"\n🚀 Aggressive Configuration Comparison")

    aggressive_config = Config()
    aggressive_config.backtesting.initial_capital = 200000  # Larger account
    aggressive_config.backtesting.commission = 0.35        # Lower commission (volume discount)
    aggressive_config.backtesting.max_position_size = 20   # Larger positions
    aggressive_config.recommendation.confidence_threshold = 0.5  # Lower threshold

    ai_aggressive = OptionsAI(config=aggressive_config)

    print("Aggressive Trading Configuration:")
    print(f"  Initial Capital: ${ai_aggressive.config.backtesting.initial_capital:,.0f}")
    print(f"  Commission: ${ai_aggressive.config.backtesting.commission}")
    print(f"  Max Position Size: {ai_aggressive.config.backtesting.max_position_size} contracts")
    print(f"  Confidence Threshold: {ai_aggressive.config.recommendation.confidence_threshold:.1%}")

    try:
        print("\n⏳ Running aggressive backtest...")
        aggressive_results = await ai_aggressive.run_backtest(backtest_config)

        if aggressive_results:
            print("✅ Aggressive backtest results:")
            print(f"  Total Return: {aggressive_results.get('total_return', 0.0):.2%}")
            print(f"  Max Drawdown: {aggressive_results.get('max_drawdown', 0.0):.2%}")
            print(f"  Win Rate: {aggressive_results.get('win_rate', 0.0):.1%}")
            print(f"  Total Trades: {aggressive_results.get('total_trades', 0)}")

            # Compare approaches
            conservative_return = results.get('total_return', 0.0) if 'results' in locals() else 0.0
            aggressive_return = aggressive_results.get('total_return', 0.0)

            print(f"\n📈 Configuration Impact:")
            print(f"  Conservative Return: {conservative_return:.2%}")
            print(f"  Aggressive Return: {aggressive_return:.2%}")

            if aggressive_return > conservative_return * 1.5:
                print("  🚀 Aggressive approach shows significantly higher returns")
            elif conservative_return > 0 and aggressive_return / conservative_return < 0.8:
                print("  🛡️ Conservative approach provides better risk-adjusted returns")
            else:
                print("  ⚖️ Both approaches show similar performance")

    except Exception as e:
        print(f"❌ Aggressive backtest failed: {e}")

    print("\n✅ Custom configuration example completed!")


async def time_period_analysis_example():
    """Example analyzing performance across different time periods."""
    print("\n📅 Time Period Analysis Example")
    print("=" * 40)

    ai = OptionsAI()

    # Define different time periods to test
    periods = [
        ('2022-Q1', '2022-01-01', '2022-03-31'),
        ('2022-Q2', '2022-04-01', '2022-06-30'),
        ('2022-Q3', '2022-07-01', '2022-09-30'),
        ('2022-Q4', '2022-10-01', '2022-12-31'),
        ('2023-Full', '2023-01-01', '2023-12-31')
    ]

    strategy = 'iron_condor'
    symbol = 'SPY'

    print(f"Analyzing {strategy.replace('_', ' ').title()} performance on {symbol} across different periods:")
    print()

    period_results = []

    for period_name, start_date, end_date in periods:
        print(f"⏳ Testing {period_name} ({start_date} to {end_date})...")

        config = {
            'symbol': symbol,
            'strategy': strategy,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': 100000
        }

        try:
            results = await ai.run_backtest(config)

            if results:
                period_result = {
                    'period': period_name,
                    'start_date': start_date,
                    'end_date': end_date,
                    'total_return': results.get('total_return', 0.0),
                    'sharpe_ratio': results.get('sharpe_ratio', 0.0),
                    'max_drawdown': results.get('max_drawdown', 0.0),
                    'win_rate': results.get('win_rate', 0.0)
                }

                period_results.append(period_result)
                print(f"  ✅ Return: {period_result['total_return']:.2%}, "
                      f"Sharpe: {period_result['sharpe_ratio']:.2f}")

            else:
                print(f"  ❌ No results for {period_name}")

        except Exception as e:
            print(f"  ❌ Error testing {period_name}: {e}")

    # Analyze results across periods
    if period_results:
        print(f"\n📊 Time Period Analysis Results:")
        print("=" * 75)
        print(f"{'Period':<12} {'Return':<10} {'Sharpe':<8} {'Drawdown':<10} {'Win Rate':<10}")
        print("-" * 75)

        total_periods = len(period_results)
        profitable_periods = 0
        best_period = None
        worst_period = None

        for result in period_results:
            print(f"{result['period']:<12} "
                  f"{result['total_return']:>8.2%} "
                  f"{result['sharpe_ratio']:>7.2f} "
                  f"{result['max_drawdown']:>9.2%} "
                  f"{result['win_rate']:>8.1%}")

            # Track statistics
            if result['total_return'] > 0:
                profitable_periods += 1

            if best_period is None or result['total_return'] > best_period['total_return']:
                best_period = result

            if worst_period is None or result['total_return'] < worst_period['total_return']:
                worst_period = result

        # Summary statistics
        print(f"\n📈 Period Analysis Summary:")
        print(f"  Total periods tested: {total_periods}")
        print(f"  Profitable periods: {profitable_periods} ({profitable_periods/total_periods:.1%})")

        if best_period:
            print(f"  Best period: {best_period['period']} ({best_period['total_return']:.2%})")

        if worst_period:
            print(f"  Worst period: {worst_period['period']} ({worst_period['total_return']:.2%})")

        # Calculate average performance
        avg_return = sum(r['total_return'] for r in period_results) / len(period_results)
        avg_sharpe = sum(r['sharpe_ratio'] for r in period_results) / len(period_results)

        print(f"  Average return: {avg_return:.2%}")
        print(f"  Average Sharpe ratio: {avg_sharpe:.2f}")

        # Consistency analysis
        returns = [r['total_return'] for r in period_results]
        return_std = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5

        print(f"  Return consistency (std dev): {return_std:.2%}")

        if return_std < 0.05:
            print("  🎯 Strategy shows consistent performance across periods")
        elif return_std < 0.10:
            print("  ⚡ Strategy shows moderate consistency")
        else:
            print("  ⚠️ Strategy performance varies significantly across periods")

    print("\n✅ Time period analysis example completed!")


async def advanced_workflow_example():
    """Example of advanced backtesting workflow."""
    print("\n🎯 Advanced Workflow Example")
    print("=" * 40)
    print("This example demonstrates a complete analysis workflow:")
    print("1. Get AI recommendations")
    print("2. Backtest top recommendations")
    print("3. Compare performance")
    print("4. Generate insights")
    print()

    ai = OptionsAI()
    symbol = 'QQQ'  # NASDAQ ETF

    # Step 1: Get AI recommendations
    print(f"Step 1: Getting recommendations for {symbol}...")

    try:
        recommendations = await ai.get_recommendations(symbol)

        if not recommendations:
            print(f"❌ No recommendations found for {symbol}")
            return

        print(f"✅ Found {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['strategy'].replace('_', ' ').title()} "
                  f"(Confidence: {rec['confidence']:.1%})")

        # Step 2: Backtest top recommendations
        print(f"\nStep 2: Backtesting top recommendations...")

        backtest_results = []
        top_strategies = recommendations[:3]  # Test top 3

        for rec in top_strategies:
            strategy = rec['strategy']
            print(f"  ⏳ Backtesting {strategy.replace('_', ' ').title()}...")

            config = {
                'symbol': symbol,
                'strategy': strategy,
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'initial_capital': 100000
            }

            try:
                results = await ai.run_backtest(config)

                if results:
                    backtest_result = {
                        'strategy': strategy,
                        'ai_confidence': rec['confidence'],
                        'ai_score': rec['score'],
                        'total_return': results.get('total_return', 0.0),
                        'sharpe_ratio': results.get('sharpe_ratio', 0.0),
                        'max_drawdown': results.get('max_drawdown', 0.0),
                        'win_rate': results.get('win_rate', 0.0)
                    }

                    backtest_results.append(backtest_result)
                    print(f"    ✅ Return: {backtest_result['total_return']:.2%}")

            except Exception as e:
                print(f"    ❌ Error: {e}")

        # Step 3: Compare performance
        if backtest_results:
            print(f"\nStep 3: Performance comparison...")
            print("=" * 90)
            print(f"{'Strategy':<18} {'AI Conf':<8} {'AI Score':<9} {'Return':<8} {'Sharpe':<7} {'Drawdown':<9} {'Win Rate':<8}")
            print("-" * 90)

            # Sort by total return for comparison
            sorted_results = sorted(backtest_results, key=lambda x: x['total_return'], reverse=True)

            for result in sorted_results:
                strategy_name = result['strategy'].replace('_', ' ').title()
                print(f"{strategy_name:<18} "
                      f"{result['ai_confidence']:>6.1%} "
                      f"{result['ai_score']:>8} "
                      f"{result['total_return']:>7.2%} "
                      f"{result['sharpe_ratio']:>6.2f} "
                      f"{result['max_drawdown']:>8.2%} "
                      f"{result['win_rate']:>7.1%}")

            # Step 4: Generate insights
            print(f"\nStep 4: Analysis insights...")

            best_backtest = sorted_results[0]
            highest_ai_confidence = max(backtest_results, key=lambda x: x['ai_confidence'])

            print(f"\n🏆 Best Backtested Performance:")
            print(f"  Strategy: {best_backtest['strategy'].replace('_', ' ').title()}")
            print(f"  Return: {best_backtest['total_return']:.2%}")
            print(f"  AI Confidence: {best_backtest['ai_confidence']:.1%}")

            print(f"\n🤖 Highest AI Confidence:")
            print(f"  Strategy: {highest_ai_confidence['strategy'].replace('_', ' ').title()}")
            print(f"  AI Confidence: {highest_ai_confidence['ai_confidence']:.1%}")
            print(f"  Actual Return: {highest_ai_confidence['total_return']:.2%}")

            # AI prediction accuracy analysis
            print(f"\n📊 AI Prediction Analysis:")

            # Check correlation between AI confidence and actual performance
            confidence_vs_return = [
                (r['ai_confidence'], r['total_return']) for r in backtest_results
            ]

            if len(confidence_vs_return) >= 2:
                # Simple correlation check
                high_conf_performance = [r['total_return'] for r in backtest_results
                                       if r['ai_confidence'] >= 0.7]
                low_conf_performance = [r['total_return'] for r in backtest_results
                                      if r['ai_confidence'] < 0.7]

                if high_conf_performance and low_conf_performance:
                    avg_high_conf = sum(high_conf_performance) / len(high_conf_performance)
                    avg_low_conf = sum(low_conf_performance) / len(low_conf_performance)

                    if avg_high_conf > avg_low_conf:
                        print("  ✅ AI confidence correlates positively with performance")
                    else:
                        print("  ⚠️ AI confidence may not predict performance in this case")

                elif high_conf_performance:
                    print("  📈 All recommendations were high confidence")

            # Risk-adjusted performance insights
            best_risk_adjusted = max(backtest_results, key=lambda x: x['sharpe_ratio'])
            if best_risk_adjusted != best_backtest:
                print(f"\n⚖️ Best Risk-Adjusted Performance:")
                print(f"  Strategy: {best_risk_adjusted['strategy'].replace('_', ' ').title()}")
                print(f"  Sharpe Ratio: {best_risk_adjusted['sharpe_ratio']:.2f}")
                print(f"  Return: {best_risk_adjusted['total_return']:.2%}")

    except Exception as e:
        print(f"❌ Advanced workflow error: {e}")

    print("\n✅ Advanced workflow example completed!")


async def main():
    """Run all backtesting examples."""
    print("🏛️ ChessTrader Options AI - Backtesting Examples")
    print("=" * 60)
    print()
    print("This example demonstrates comprehensive backtesting capabilities:")
    print("• Single strategy backtesting with performance analysis")
    print("• Multi-strategy comparison and ranking")
    print("• Custom configuration for different trading styles")
    print("• Time period analysis for consistency evaluation")
    print("• Advanced workflow combining AI recommendations with backtesting")
    print()
    print("Note: Examples use mock data when real market data is unavailable.")
    print()

    try:
        # Run all example functions
        await basic_backtest_example()
        await strategy_comparison_example()
        await custom_configuration_example()
        await time_period_analysis_example()
        await advanced_workflow_example()

        print("\n🎉 All backtesting examples completed successfully!")
        print("\nNext steps:")
        print("• Experiment with different symbols and strategies")
        print("• Try the CLI backtesting: chesstrader backtest --help")
        print("• Read the documentation: docs/cli_guide.md")
        print("• Configure custom settings: docs/api_reference.md#configuration")

    except KeyboardInterrupt:
        print("\n\n⚠️ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error in examples: {e}")
        print("This may be due to missing dependencies or configuration issues.")
        print("Please check the installation and try again.")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
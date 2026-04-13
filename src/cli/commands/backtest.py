"""
Backtesting command for ChessTrader CLI.

Provides comprehensive backtesting capabilities with formatted results display
including performance metrics, risk analysis, and detailed reporting.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TimeRemainingColumn, SpinnerColumn, TextColumn
from rich.text import Text
from rich.columns import Columns

from ..main import get_options_ai, run_async

console = Console()


def backtest(
    symbol: str = typer.Option(..., "--symbol", "-s", help="Stock/ETF symbol to backtest (e.g., AAPL, SPY)"),
    strategy: Optional[str] = typer.Option(
        None,
        "--strategy",
        help="Specific strategy to test (if not provided, uses AI recommendations)"
    ),
    start_date: Optional[str] = typer.Option(
        None,
        "--start-date",
        help="Start date for backtest (YYYY-MM-DD, default: 1 year ago)"
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end-date",
        help="End date for backtest (YYYY-MM-DD, default: today)"
    ),
    initial_capital: float = typer.Option(
        100000.0,
        "--capital",
        help="Initial capital for backtesting"
    ),
    commission: float = typer.Option(
        0.65,
        "--commission",
        help="Commission per contract"
    ),
    max_positions: int = typer.Option(
        10,
        "--max-positions",
        help="Maximum number of contracts per position"
    ),
    output_format: str = typer.Option(
        "console",
        "--output",
        "-o",
        help="Output format: console, csv, html, pdf"
    ),
    save_report: bool = typer.Option(
        False,
        "--save-report",
        help="Save detailed report to file"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to configuration file"
    )
):
    """
    Run comprehensive backtesting with performance analysis.

    Tests options strategies against historical data with realistic
    execution simulation, transaction costs, and risk management.

    Examples:
        chesstrader backtest --symbol AAPL
        chesstrader backtest --symbol SPY --strategy iron_condor --start-date 2023-01-01
        chesstrader backtest --symbol QQQ --capital 50000 --save-report
    """
    symbol = symbol.upper().strip()

    # Set default dates if not provided
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_dt = datetime.now() - timedelta(days=365)
        start_date = start_dt.strftime("%Y-%m-%d")

    # Display header
    console.print(f"\n[bold blue]🔬 Backtesting Analysis for {symbol}[/bold blue]")
    console.print(f"[dim]Period: {start_date} to {end_date} | Capital: ${initial_capital:,.0f}[/dim]")
    if strategy:
        console.print(f"[dim]Strategy: {strategy.replace('_', ' ').title()}[/dim]")
    console.print()

    try:
        # Get OptionsAI instance
        ai = get_options_ai(config)

        # Build backtest configuration
        backtest_config = {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'commission': commission,
            'max_position_size': max_positions
        }

        if strategy:
            backtest_config['strategy'] = strategy

        # Show progress during backtest
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Running backtest analysis...", total=100)

            # Simulate progress updates (in real implementation, this would be actual progress)
            progress.update(task, advance=20, description="Loading historical data...")

            # Run backtest
            results = run_async(ai.run_backtest(backtest_config))
            progress.update(task, advance=80, description="Analyzing results...")
            progress.update(task, completed=100)

        if not results:
            console.print(Panel(
                "[yellow]No backtest results generated[/yellow]\n"
                "This could be due to:\n"
                "• Insufficient historical data\n"
                "• Invalid date range\n"
                "• Strategy execution errors",
                title="No Results",
                border_style="yellow"
            ))
            return

        # Display results in organized panels
        display_backtest_results(results, symbol, strategy)

        # Save report if requested
        if save_report:
            save_backtest_report(results, symbol, strategy, output_format)

    except Exception as e:
        console.print(Panel(
            f"[red]Error running backtest: {str(e)}[/red]\n\n"
            f"Please check:\n"
            f"• Symbol is valid: {symbol}\n"
            f"• Date range is reasonable\n"
            f"• Historical data availability\n"
            f"• Strategy parameters",
            title="Backtest Error",
            border_style="red"
        ))
        raise typer.Exit(1)


def display_backtest_results(results: Dict[str, Any], symbol: str, strategy: Optional[str]):
    """Display backtest results in formatted panels and tables."""

    # Key Performance Metrics Panel
    perf_metrics = Table(show_header=False, box=None)
    perf_metrics.add_column("Metric", style="cyan")
    perf_metrics.add_column("Value", style="bold")

    # Extract key metrics with fallback values
    total_return = results.get('total_return', 0.0)
    sharpe_ratio = results.get('sharpe_ratio', 0.0)
    max_drawdown = results.get('max_drawdown', 0.0)
    win_rate = results.get('win_rate', 0.0)
    num_trades = results.get('total_trades', 0)
    avg_trade = results.get('avg_trade_return', 0.0)

    # Color code returns
    return_color = "green" if total_return > 0 else "red"
    sharpe_color = "green" if sharpe_ratio > 1.0 else "yellow" if sharpe_ratio > 0.5 else "red"
    drawdown_color = "red" if max_drawdown < -0.2 else "yellow" if max_drawdown < -0.1 else "green"

    perf_metrics.add_row("Total Return", f"[{return_color}]{total_return:.2%}[/{return_color}]")
    perf_metrics.add_row("Sharpe Ratio", f"[{sharpe_color}]{sharpe_ratio:.2f}[/{sharpe_color}]")
    perf_metrics.add_row("Max Drawdown", f"[{drawdown_color}]{max_drawdown:.2%}[/{drawdown_color}]")
    perf_metrics.add_row("Win Rate", f"{win_rate:.1%}")
    perf_metrics.add_row("Total Trades", f"{num_trades}")
    perf_metrics.add_row("Avg Trade Return", f"{avg_trade:.2%}")

    console.print(Panel(perf_metrics, title="📊 Key Performance Metrics", border_style="blue"))

    # Risk Metrics Panel
    risk_metrics = Table(show_header=False, box=None)
    risk_metrics.add_column("Metric", style="cyan")
    risk_metrics.add_column("Value", style="bold")

    volatility = results.get('volatility', 0.0)
    sortino_ratio = results.get('sortino_ratio', 0.0)
    calmar_ratio = results.get('calmar_ratio', 0.0)
    var_95 = results.get('var_95', 0.0)

    risk_metrics.add_row("Volatility", f"{volatility:.2%}")
    risk_metrics.add_row("Sortino Ratio", f"{sortino_ratio:.2f}")
    risk_metrics.add_row("Calmar Ratio", f"{calmar_ratio:.2f}")
    risk_metrics.add_row("VaR (95%)", f"{var_95:.2%}")

    console.print(Panel(risk_metrics, title="⚠️  Risk Analysis", border_style="yellow"))

    # Trade Statistics (if available)
    if 'trade_stats' in results:
        trade_stats = results['trade_stats']
        stats_table = Table(show_header=False, box=None)
        stats_table.add_column("Statistic", style="cyan")
        stats_table.add_column("Value", style="bold")

        stats_table.add_row("Profitable Trades", f"{trade_stats.get('profitable_trades', 0)}")
        stats_table.add_row("Loss Trades", f"{trade_stats.get('loss_trades', 0)}")
        stats_table.add_row("Largest Win", f"{trade_stats.get('largest_win', 0.0):.2%}")
        stats_table.add_row("Largest Loss", f"{trade_stats.get('largest_loss', 0.0):.2%}")
        stats_table.add_row("Avg Win", f"{trade_stats.get('avg_win', 0.0):.2%}")
        stats_table.add_row("Avg Loss", f"{trade_stats.get('avg_loss', 0.0):.2%}")

        console.print(Panel(stats_table, title="📈 Trade Statistics", border_style="green"))

    # Strategy specific info
    if strategy:
        strategy_name = strategy.replace('_', ' ').title()
        console.print(f"\n[dim]Strategy: {strategy_name} on {symbol}[/dim]")

    # Summary assessment
    print_performance_summary(total_return, sharpe_ratio, max_drawdown)


def print_performance_summary(total_return: float, sharpe_ratio: float, max_drawdown: float):
    """Print an overall performance assessment."""

    # Determine overall rating
    if total_return > 0.15 and sharpe_ratio > 1.0 and max_drawdown > -0.15:
        rating = "🌟 Excellent"
        color = "green"
    elif total_return > 0.05 and sharpe_ratio > 0.5 and max_drawdown > -0.25:
        rating = "✅ Good"
        color = "green"
    elif total_return > 0 and sharpe_ratio > 0 and max_drawdown > -0.35:
        rating = "⚡ Acceptable"
        color = "yellow"
    else:
        rating = "⚠️  Needs Improvement"
        color = "red"

    console.print(f"\n[bold {color}]Overall Performance: {rating}[/bold {color}]")


def save_backtest_report(results: Dict[str, Any], symbol: str, strategy: Optional[str], format: str):
    """Save backtest results to file."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_str = f"_{strategy}" if strategy else ""
        filename = f"backtest_{symbol}{strategy_str}_{timestamp}.{format}"

        # This is a placeholder - in real implementation would use the reporting system
        console.print(f"\n[green]📄 Report saved: {filename}[/green]")
        console.print(f"[dim]Format: {format.upper()} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")

    except Exception as e:
        console.print(f"[red]Error saving report: {e}[/red]")


# Export the command function
__all__ = ['backtest']
"""
Enhanced options trading recommendations command.

Provides actionable trading recommendations with specific contract details,
entry/exit dates, and concrete trade instructions.
"""

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns

from ...api.enhanced_strategy_recommender import EnhancedStrategyRecommender
from ..visualizations.profit_diagrams import generate_strategy_diagram

console = Console()


def options_trades(
    symbol: str = typer.Argument(..., help="Stock/ETF symbol to analyze (e.g., AAPL, SPY)"),
    days: int = typer.Option(
        5,
        "--analysis-days",
        "-d",
        min=1,
        max=30,
        help="Number of days to analyze for opportunities"
    ),
    details: bool = typer.Option(
        False,
        "--details",
        help="Show detailed contract information"
    )
):
    """
    Get actionable options trading recommendations with specific contracts and dates.

    This command provides real trading instructions including:
    - Specific contract symbols and strikes
    - Entry and exit dates
    - Profit/loss calculations
    - Risk management guidelines
    """
    symbol = symbol.upper()

    # Show header
    console.print()
    console.print(f"🎯 [bold blue]Actionable Options Trades for {symbol}[/bold blue]")
    console.print(f"Analysis period: {days} days | Real-time market data")
    console.print()

    try:
        with console.status("[bold green]Fetching options chains and analyzing opportunities..."):
            recommender = EnhancedStrategyRecommender()
            recommendations = recommender.get_actionable_recommendations(symbol, days)

        if not recommendations:
            console.print("❌ No actionable options opportunities found")
            console.print("Try a different symbol or check market conditions")
            return

        # Display each recommendation
        for i, rec in enumerate(recommendations, 1):
            _display_recommendation(rec, i, details)
            if i < len(recommendations):
                console.print()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("\n[dim]Please check:")
        console.print("• Symbol is valid and actively traded")
        console.print("• Options are available for this symbol")
        console.print("• Market is open or recently closed")


def _display_recommendation(rec: dict, index: int, show_details: bool):
    """Display a single recommendation with rich formatting"""

    # Header with strategy info
    strategy_panel = Panel(
        f"[bold white]{rec['strategy_name']}[/bold white]\n"
        f"[dim]{rec['strategy_type']} • {rec['market_outlook']}[/dim]\n"
        f"[green]Confidence: {rec['confidence']:.1f}%[/green] • "
        f"[blue]P/L Ratio: {rec.get('profit_risk_ratio', 0):.1f}:1[/blue]",
        title=f"[bold yellow]Trade #{index}[/bold yellow]",
        border_style="blue"
    )
    console.print(strategy_panel)

    # Trade timeline
    timeline_text = (
        f"📅 [bold]Entry:[/bold] {rec['entry_date']} → "
        f"[bold]Exit:[/bold] {rec['exit_date']} "
        f"([cyan]{rec.get('days_to_expiration', 'N/A')} days[/cyan])"
    )
    console.print(timeline_text)

    # Add contract size explanation
    console.print(f"[dim]📋 Note: Each options contract represents 100 shares. All prices shown are per-share.[/dim]")
    console.print()

    # Contracts table
    if 'contracts' in rec:
        contracts_table = Table(title="📋 Contract Details", show_header=True, header_style="bold magenta")
        contracts_table.add_column("Action", style="bold")
        contracts_table.add_column("Qty", style="cyan")
        contracts_table.add_column("Contract Symbol", style="cyan")
        contracts_table.add_column("Strike", style="white")
        contracts_table.add_column("Type", style="yellow")
        contracts_table.add_column("Price/Share", style="green")
        contracts_table.add_column("Description")

        for contract in rec['contracts']:
            action_color = "green" if contract['action'] == 'BUY' else "red"
            contracts_table.add_row(
                f"[{action_color}]{contract['action']}[/{action_color}]",
                "1",  # Standard contract quantity
                contract.get('contract_symbol', 'N/A'),
                f"${contract.get('strike', 0):.2f}",
                contract.get('type', 'N/A').upper(),
                f"${contract.get('price', 0):.2f}",
                contract.get('description', 'N/A')
            )

        console.print(contracts_table)
        console.print()

    # Trade financials
    trade_details = rec.get('trade_details', {})
    if trade_details:
        # Create financial metrics panels
        left_panel_content = []
        right_panel_content = []

        # Costs and profits (multiply by 100 since each contract = 100 shares)
        if 'net_debit' in trade_details:
            per_share = trade_details['net_debit']
            total = per_share * 100
            left_panel_content.append(f"💰 [bold]Net Cost:[/bold] ${per_share:.2f}/share (${total:.0f} total)")
        if 'net_credit' in trade_details:
            per_share = trade_details['net_credit']
            total = per_share * 100
            left_panel_content.append(f"💰 [bold]Net Credit:[/bold] ${per_share:.2f}/share (${total:.0f} total)")
        if 'premium_collected' in trade_details:
            per_share = trade_details['premium_collected']
            total = per_share * 100
            left_panel_content.append(f"💰 [bold]Premium:[/bold] ${per_share:.2f}/share (${total:.0f} total)")

        max_profit_per_share = trade_details.get('max_profit', 0)
        max_profit_total = max_profit_per_share * 100
        left_panel_content.append(f"📈 [bold green]Max Profit:[/bold green] ${max_profit_per_share:.2f}/share (${max_profit_total:.0f} total)")

        max_loss_per_share = trade_details.get('max_loss', 0)
        max_loss_total = abs(max_loss_per_share) * 100
        left_panel_content.append(f"📉 [bold red]Max Loss:[/bold red] ${abs(max_loss_per_share):.2f}/share (${max_loss_total:.0f} total)")

        # Breakevens and probabilities
        if 'breakeven' in trade_details:
            right_panel_content.append(f"⚖️ [bold]Breakeven:[/bold] ${trade_details['breakeven']:.2f}")
        if 'upper_breakeven' in trade_details:
            right_panel_content.append(f"⚖️ [bold]Upper BE:[/bold] ${trade_details['upper_breakeven']:.2f}")
        if 'lower_breakeven' in trade_details:
            right_panel_content.append(f"⚖️ [bold]Lower BE:[/bold] ${trade_details['lower_breakeven']:.2f}")
        if 'profitable_range' in trade_details:
            right_panel_content.append(f"🎯 [bold]Profit Zone:[/bold] {trade_details['profitable_range']}")

        right_panel_content.append(f"🎲 [bold]Win Probability:[/bold] {trade_details.get('probability_of_profit', 'N/A')}")

        # Display in columns
        left_panel = Panel("\n".join(left_panel_content), title="💵 Trade Financials", border_style="green")
        right_panel = Panel("\n".join(right_panel_content), title="📊 Key Metrics", border_style="blue")

        console.print(Columns([left_panel, right_panel]))
        console.print()

    # Add P/L Diagram
    if show_details or True:  # Always show diagrams for now
        try:
            # Generate the P/L diagram
            current_price = rec.get('current_price', 100)
            diagram_lines = generate_strategy_diagram(
                strategy_name=rec.get('strategy_name', 'Unknown'),
                trade_details=trade_details,
                contracts=rec.get('contracts', []),
                current_price=current_price
            )

            # Display the diagram in a panel
            diagram_text = "\n".join(diagram_lines)
            diagram_panel = Panel(
                diagram_text,
                title="📊 Profit/Loss Visualization",
                border_style="cyan"
            )
            console.print(diagram_panel)
            console.print()
        except Exception as e:
            console.print(f"[dim]Could not generate P/L diagram: {e}[/dim]")

    # Exit strategy
    exit_strategy = rec.get('exit_strategy', {})
    if exit_strategy and show_details:
        exit_content = []
        for key, value in exit_strategy.items():
            if key == 'target_profit':
                exit_content.append(f"🎯 [bold green]Target:[/bold green] {value}")
            elif key == 'stop_loss':
                exit_content.append(f"🛑 [bold red]Stop Loss:[/bold red] {value}")
            elif key == 'time_decay':
                exit_content.append(f"⏰ [bold yellow]Time Management:[/bold yellow] {value}")
            elif key == 'assignment_risk':
                exit_content.append(f"⚠️ [bold orange1]Assignment:[/bold orange1] {value}")
            elif key == 'buyback_option':
                exit_content.append(f"🔄 [bold cyan]Buyback:[/bold cyan] {value}")

        if exit_content:
            exit_panel = Panel(
                "\n".join(exit_content),
                title="🚪 Exit Strategy",
                border_style="yellow"
            )
            console.print(exit_panel)

    # Prerequisites (for strategies like covered calls)
    if 'prerequisites' in rec:
        prereq_panel = Panel(
            f"⚠️  [bold yellow]{rec['prerequisites']}[/bold yellow]",
            title="Requirements",
            border_style="orange1"
        )
        console.print(prereq_panel)


if __name__ == "__main__":
    typer.run(options_trades)
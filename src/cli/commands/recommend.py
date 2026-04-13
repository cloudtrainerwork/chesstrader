"""
Strategy recommendation command for ChessTrader CLI.

Provides formatted display of AI-generated options strategy recommendations
with confidence scores and detailed analysis.
"""

from typing import Optional, List, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..main import get_options_ai, run_async

console = Console()


def recommend(
    symbol: str = typer.Argument(..., help="Stock/ETF symbol to analyze (e.g., AAPL, SPY)"),
    confidence: float = typer.Option(
        0.4,
        "--confidence",
        "-c",
        min=0.0,
        max=1.0,
        help="Minimum confidence threshold for recommendations (0.0-1.0)"
    ),
    max_results: int = typer.Option(
        3,
        "--max-results",
        "-n",
        min=1,
        max=10,
        help="Maximum number of recommendations to display"
    ),
    details: bool = typer.Option(
        False,
        "--details",
        "-d",
        help="Show detailed strategy information"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to configuration file"
    )
):
    """
    Get AI-powered options strategy recommendations for a symbol.

    Analyzes market conditions, volatility patterns, and regime detection
    to recommend optimal options strategies with confidence scores.

    Examples:
        chesstrader recommend AAPL
        chesstrader recommend SPY --confidence 0.6 --max-results 5
        chesstrader recommend QQQ --details
    """
    symbol = symbol.upper().strip()

    # Display header
    console.print(f"\n[bold blue]🎯 Strategy Recommendations for {symbol}[/bold blue]")
    console.print(f"[dim]Confidence threshold: {confidence:.1%} | Max results: {max_results}[/dim]\n")

    try:
        # Get OptionsAI instance
        ai = get_options_ai(config)

        # Update configuration if needed
        if confidence != 0.4 or max_results != 3:
            ai.update_config(
                recommendation__confidence_threshold=confidence,
                recommendation__max_recommendations=max_results
            )

        # Show loading spinner while getting recommendations
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing market conditions and generating recommendations...", total=None)

            # Get recommendations
            recommendations = run_async(ai.get_recommendations(symbol))
            progress.update(task, completed=True)

        if not recommendations:
            console.print(Panel(
                f"[yellow]No recommendations found for {symbol}[/yellow]\n"
                f"This could be due to:\n"
                f"• Confidence threshold too high ({confidence:.1%})\n"
                f"• Insufficient market data\n"
                f"• Market conditions outside model parameters",
                title="No Recommendations",
                border_style="yellow"
            ))
            return

        # Create recommendations table
        table = Table(
            title=f"Options Strategy Recommendations - {symbol}",
            show_header=True,
            header_style="bold blue",
            border_style="blue",
            title_style="bold blue"
        )

        # Add columns
        table.add_column("Strategy", style="cyan", no_wrap=True)
        table.add_column("Confidence", style="bold", justify="center")
        table.add_column("Score", justify="center")
        table.add_column("Market Outlook", style="italic")

        if details:
            table.add_column("Risk Level", justify="center")
            table.add_column("Time Frame", justify="center")

        # Add rows with color coding based on confidence
        for rec in recommendations[:max_results]:
            strategy = rec.get('strategy', 'Unknown')
            conf = rec.get('confidence', 0.0)
            score = rec.get('score', 0)
            outlook = rec.get('market_outlook', 'N/A')

            # Color code confidence
            if conf >= 0.8:
                conf_text = f"[green]{conf:.1%}[/green]"
            elif conf >= 0.6:
                conf_text = f"[yellow]{conf:.1%}[/yellow]"
            else:
                conf_text = f"[red]{conf:.1%}[/red]"

            # Format score
            score_text = f"{score}" if isinstance(score, int) else f"{score:.1f}"

            row = [strategy.replace('_', ' ').title(), conf_text, score_text, outlook]

            if details:
                # Get additional details if requested
                try:
                    strategy_details = ai.get_strategy_details(strategy)
                    risk_level = strategy_details.get('risk_profile', 'Unknown')
                    time_frame = strategy_details.get('typical_duration', 'N/A')
                    row.extend([risk_level.title(), time_frame])
                except:
                    row.extend(['Unknown', 'N/A'])

            table.add_row(*row)

        console.print(table)

        # Show additional info if details requested
        if details and recommendations:
            console.print(f"\n[dim]💡 Strategy Details:[/dim]")
            for i, rec in enumerate(recommendations[:max_results], 1):
                strategy = rec.get('strategy', 'Unknown')
                try:
                    details_info = ai.get_strategy_details(strategy)
                    description = details_info.get('description', 'No description available')
                    console.print(f"{i}. [cyan]{strategy.replace('_', ' ').title()}[/cyan]: {description}")
                except:
                    console.print(f"{i}. [cyan]{strategy.replace('_', ' ').title()}[/cyan]: Details not available")

        # Show summary statistics
        if recommendations:
            avg_confidence = sum(r.get('confidence', 0) for r in recommendations) / len(recommendations)
            high_conf_count = sum(1 for r in recommendations if r.get('confidence', 0) >= 0.7)

            console.print(f"\n[dim]📊 Summary: {len(recommendations)} recommendations found | "
                         f"Average confidence: {avg_confidence:.1%} | "
                         f"High confidence (≥70%): {high_conf_count}[/dim]")

    except Exception as e:
        console.print(Panel(
            f"[red]Error getting recommendations: {str(e)}[/red]\n\n"
            f"Please check:\n"
            f"• Symbol is valid: {symbol}\n"
            f"• Network connectivity\n"
            f"• Configuration settings",
            title="Recommendation Error",
            border_style="red"
        ))
        raise typer.Exit(1)


# Export the command function
__all__ = ['recommend']
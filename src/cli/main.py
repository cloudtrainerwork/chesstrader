"""
Main CLI application for ChessTrader Options AI.

Provides command-line interface for strategy recommendations and backtesting
using Typer for modern CLI experience with Rich formatting.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Import OptionsAI
_options_ai_import_error: Optional[Exception] = None
try:
    # Try relative import first (when installed as package)
    from ..main import OptionsAI
except (ImportError, ValueError):
    # Fallback for development/standalone usage
    try:
        import sys
        import os
        from pathlib import Path

        # Add parent directory to path
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from main import OptionsAI
    except ImportError as e:
        # Defer reporting to command execution. Calling sys.exit() at import
        # time aborts the whole process — including pytest collection — for any
        # module that merely imports this one.
        OptionsAI = None
        _options_ai_import_error = e

# Create Typer app
app = typer.Typer(
    name="chesstrader",
    help="ChessTrader Options AI - Intelligent options strategy recommendations and backtesting",
    add_completion=False,
    rich_markup_mode="rich",
)

# Rich console for formatted output
console = Console()

# Global OptionsAI instance
_options_ai: Optional["OptionsAI"] = None


def get_options_ai(config_path: Optional[str] = None) -> "OptionsAI":
    """Get or create OptionsAI instance."""
    global _options_ai
    if OptionsAI is None:
        console.print(
            f"[red]OptionsAI is unavailable: {_options_ai_import_error}[/red]"
        )
        console.print(
            "Please ensure all dependencies are installed and the project is "
            "properly configured."
        )
        raise typer.Exit(1)
    if _options_ai is None:
        try:
            _options_ai = OptionsAI(config_path=config_path)
        except Exception as e:
            console.print(f"[red]Error initializing OptionsAI: {e}[/red]")
            raise typer.Exit(1)
    return _options_ai


@app.callback()
def main(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file (JSON format)",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output"
    ),
    version: bool = typer.Option(
        False,
        "--version",
        help="Show version and exit"
    )
):
    """
    ChessTrader Options AI - Intelligent options strategy recommendations and backtesting.

    A sophisticated AI system that combines chess-inspired neural networks with
    reinforcement learning to provide intelligent options trading strategies.
    """
    if version:
        ai = get_options_ai(config)
        version_text = Text(f"ChessTrader Options AI v{ai.version()}", style="bold blue")
        console.print(Panel(version_text, title="Version Info", border_style="blue"))
        raise typer.Exit()

    # Initialize OptionsAI with config
    get_options_ai(config)

    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")


def run_async(coro):
    """Helper to run async functions in CLI commands."""
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a new one for CLI
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create a new one
        return asyncio.run(coro)


# Version command
@app.command()
def version():
    """Show version information."""
    ai = get_options_ai()
    version_text = Text(f"ChessTrader Options AI v{ai.version()}", style="bold blue")
    features = [
        "✨ Chess-inspired neural architecture",
        "🧠 Reinforcement learning position management",
        "📊 Comprehensive backtesting engine",
        "🎯 Intelligent strategy recommendations",
        "📈 Monte Carlo risk analysis"
    ]

    console.print(Panel(
        version_text,
        title="ChessTrader Options AI",
        subtitle="Advanced Options Trading Intelligence",
        border_style="blue"
    ))

    for feature in features:
        console.print(f"  {feature}")


# Import and register commands
from .commands.recommend import recommend
from .commands.backtest import backtest
from .commands.options_trades import options_trades

# Add commands to app
app.command(name="recommend", help="Get AI-powered strategy recommendations")(recommend)
app.command(name="backtest", help="Run comprehensive backtesting analysis")(backtest)
app.command(name="trades", help="Get actionable options trading recommendations with specific contracts")(options_trades)


if __name__ == "__main__":
    app()
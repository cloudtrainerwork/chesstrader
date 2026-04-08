"""
Command-line interface for complete backtesting workflow orchestration

Main entry point for running backtests, walk-forward optimization,
and Monte Carlo analysis with configuration and output management.
"""

import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
import os
import json
import sys
from pathlib import Path

# Import backtesting components
from src.backtesting.core.engine import BacktestEngine
from src.backtesting.optimization.walk_forward import WalkForwardOptimizer
from src.backtesting.monte_carlo.simulator import MonteCarloSimulator
from src.backtesting.monte_carlo.analysis import StatisticalAnalyzer
from src.backtesting.performance.metrics import PerformanceCalculator
from src.backtesting.performance.tearsheet import TearsheetGenerator
from src.backtesting.performance.reporting import ReportGenerator

logger = logging.getLogger(__name__)


class BacktestCLI:
    """
    Command-line interface for backtesting system

    Orchestrates complete backtesting workflows with configuration
    management and professional output generation.
    """

    def __init__(self):
        """Initialize CLI interface"""
        self.setup_logging()
        self.performance_calculator = PerformanceCalculator()
        self.tearsheet_generator = TearsheetGenerator()
        self.report_generator = ReportGenerator()
        self.statistical_analyzer = StatisticalAnalyzer()

    def setup_logging(self, log_level: str = 'INFO'):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )

    def parse_arguments(self) -> Dict[str, Any]:
        """Parse command-line arguments"""
        parser = argparse.ArgumentParser(
            description='ChessTrader Backtesting Engine CLI',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # Required arguments
        parser.add_argument(
            '--start-date',
            required=True,
            help='Backtest start date (YYYY-MM-DD)'
        )
        parser.add_argument(
            '--end-date',
            required=True,
            help='Backtest end date (YYYY-MM-DD)'
        )
        parser.add_argument(
            '--strategy',
            required=True,
            choices=['iron_condor', 'iron_butterfly', 'straddle', 'strangle'],
            help='Options strategy to backtest'
        )
        parser.add_argument(
            '--symbol',
            required=True,
            help='Symbol to trade (e.g., SPY, QQQ)'
        )

        # Optional arguments
        parser.add_argument(
            '--output-dir',
            default='./backtest_results',
            help='Output directory for results'
        )
        parser.add_argument(
            '--n-simulations',
            type=int,
            default=1000,
            help='Number of Monte Carlo simulations'
        )
        parser.add_argument(
            '--walk-forward',
            action='store_true',
            help='Run walk-forward optimization'
        )
        parser.add_argument(
            '--monte-carlo',
            action='store_true',
            help='Run Monte Carlo analysis'
        )
        parser.add_argument(
            '--config-file',
            help='JSON configuration file'
        )
        parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO',
            help='Logging level'
        )

        args = parser.parse_args()

        # Convert to dictionary
        config = vars(args)

        # Load configuration file if provided
        if config.get('config_file'):
            config.update(self.load_config_file(config['config_file']))

        return config

    def load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
            return {}

    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters"""
        try:
            # Required fields
            required_fields = ['start_date', 'end_date', 'strategy', 'symbol']
            for field in required_fields:
                if field not in config or not config[field]:
                    logger.error(f"Missing required field: {field}")
                    return False

            # Date validation
            start_date = datetime.strptime(config['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(config['end_date'], '%Y-%m-%d')

            if start_date >= end_date:
                logger.error("Start date must be before end date")
                return False

            if end_date > datetime.now():
                logger.error("End date cannot be in the future")
                return False

            # Simulation count validation
            if config.get('n_simulations', 0) <= 0:
                logger.error("Number of simulations must be positive")
                return False

            return True

        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False

    def run_complete_workflow(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run complete backtesting workflow"""
        try:
            logger.info("Starting complete backtesting workflow")

            if not self.validate_configuration(config):
                return None

            results = {}

            # Step 1: Basic backtest
            logger.info("Running basic backtest...")
            backtest_results = self.run_backtest(config)
            if not backtest_results:
                return None
            results['backtest_results'] = backtest_results

            # Step 2: Walk-forward optimization (if requested)
            if config.get('walk_forward'):
                logger.info("Running walk-forward optimization...")
                wf_results = self.run_walk_forward(config)
                if wf_results is not None:
                    results['walk_forward_results'] = wf_results

            # Step 3: Monte Carlo analysis (if requested)
            if config.get('monte_carlo'):
                logger.info("Running Monte Carlo analysis...")
                mc_results = self.run_monte_carlo(config, backtest_results.get('portfolio'))
                if mc_results is not None:
                    results['monte_carlo_results'] = mc_results

            # Step 4: Generate reports
            logger.info("Generating reports...")
            report_paths = self.generate_reports(results, config)
            results['report_paths'] = report_paths

            logger.info("Complete workflow finished successfully")
            return results

        except Exception as e:
            logger.error(f"Error in complete workflow: {e}")
            return None

    def run_backtest(self, config: Dict[str, Any], progress_callback: Optional[Callable] = None) -> Optional[Dict[str, Any]]:
        """Run basic backtest"""
        try:
            if progress_callback:
                progress_callback("Initializing backtest engine...")

            # Initialize backtest engine
            engine = BacktestEngine(
                start_date=config['start_date'],
                end_date=config['end_date'],
                initial_capital=config.get('initial_capital', 100000)
            )

            if progress_callback:
                progress_callback("Loading market data...")

            # Load data and run backtest
            if progress_callback:
                progress_callback("Running backtest...")

            results = engine.run_backtest(
                symbol=config['symbol'],
                strategy_name=config['strategy'],
                strategy_params=config.get('strategy_params', {})
            )

            if progress_callback:
                progress_callback("Calculating performance metrics...")

            # Calculate performance metrics
            portfolio = results.get('portfolio')
            if portfolio:
                equity_curve = portfolio.get_equity_curve()
                trade_history = portfolio.get_trade_history()

                metrics = self.performance_calculator.calculate_all_metrics(
                    equity_curve, trade_history
                )
                results['performance_metrics'] = metrics

            logger.info("Backtest completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return None

    def run_walk_forward(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Run walk-forward optimization"""
        try:
            optimizer = WalkForwardOptimizer(
                lookback_period=config.get('lookback_period', 252),
                step_size=config.get('step_size', 63)
            )

            # Define parameter grid for optimization
            param_grid = config.get('param_grid', {
                'dte_range': [(30, 45), (20, 40), (15, 35)],
                'delta_range': [(0.15, 0.25), (0.10, 0.30)],
                'profit_target': [0.5, 0.75, 1.0]
            })

            # Load market data for optimization period
            from src.data.providers.yahoo_provider import YahooDataProvider
            data_provider = YahooDataProvider()

            market_data = data_provider.fetch_options_data(
                symbol=config['symbol'],
                start_date=config['start_date'],
                end_date=config['end_date']
            )

            results = optimizer.optimize(market_data, param_grid)
            logger.info("Walk-forward optimization completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error in walk-forward optimization: {e}")
            return None

    def run_monte_carlo(self, config: Dict[str, Any], portfolio: Any) -> Optional[pd.DataFrame]:
        """Run Monte Carlo simulation"""
        try:
            simulator = MonteCarloSimulator(
                n_simulations=config.get('n_simulations', 1000),
                random_seed=config.get('random_seed', 42)
            )

            # Run simulations
            results = simulator.simulate_backtests(
                strategy_config={
                    'strategy': config['strategy'],
                    'symbol': config['symbol'],
                    'params': config.get('strategy_params', {})
                },
                portfolio_state=portfolio
            )

            logger.info(f"Monte Carlo simulation completed with {len(results)} simulations")
            return results

        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {e}")
            return None

    def generate_reports(self, results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, str]:
        """Generate all output reports"""
        try:
            output_dir = Path(config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)

            report_paths = {}

            # Basic performance metrics
            backtest_results = results.get('backtest_results', {})
            performance_metrics = backtest_results.get('performance_metrics', {})

            if performance_metrics:
                csv_path = output_dir / 'performance_metrics.csv'
                self.save_metrics_to_csv(performance_metrics, str(csv_path))
                report_paths['metrics_csv'] = str(csv_path)

            # Portfolio equity curve
            portfolio = backtest_results.get('portfolio')
            if portfolio:
                equity_curve = portfolio.get_equity_curve()

                # Generate tearsheet
                monte_carlo_results = results.get('monte_carlo_results')
                confidence_intervals = None

                if monte_carlo_results is not None:
                    confidence_intervals = self.statistical_analyzer.calculate_confidence_intervals(
                        monte_carlo_results
                    )

                tearsheet_path = self.tearsheet_generator.generate_full_report(
                    equity_curve=equity_curve,
                    performance_metrics=performance_metrics,
                    monte_carlo_results=monte_carlo_results,
                    confidence_intervals=confidence_intervals,
                    output_path=str(output_dir / 'tearsheet')
                )

                if tearsheet_path:
                    report_paths['tearsheet'] = tearsheet_path

                # Generate HTML report
                tearsheet_data = {
                    'summary_stats': {'formatted_stats': performance_metrics},
                    'monte_carlo_analysis': {}
                }

                if monte_carlo_results is not None:
                    tearsheet_data['monte_carlo_analysis'] = {
                        'risk_metrics': self.statistical_analyzer.risk_analysis(monte_carlo_results)
                    }

                html_content = self.report_generator.generate_html_report(tearsheet_data)
                html_path = output_dir / 'report.html'

                with open(html_path, 'w') as f:
                    f.write(html_content)
                report_paths['html_report'] = str(html_path)

            # Walk-forward results
            if 'walk_forward_results' in results:
                wf_path = output_dir / 'walk_forward_results.csv'
                results['walk_forward_results'].to_csv(wf_path, index=False)
                report_paths['walk_forward_csv'] = str(wf_path)

            # Monte Carlo results
            if 'monte_carlo_results' in results:
                mc_path = output_dir / 'monte_carlo_results.csv'
                self.report_generator.export_monte_carlo_results(
                    results['monte_carlo_results'], str(mc_path)
                )
                report_paths['monte_carlo_csv'] = str(mc_path)

            logger.info(f"Reports generated in {output_dir}")
            return report_paths

        except Exception as e:
            logger.error(f"Error generating reports: {e}")
            return {}

    def save_metrics_to_csv(self, metrics: Dict[str, Any], file_path: str):
        """Save performance metrics to CSV file"""
        try:
            df = pd.DataFrame([metrics])
            df.to_csv(file_path, index=False)
            logger.info(f"Metrics saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving metrics to CSV: {e}")

    def generate_output_files(self, results: Dict[str, Any], config: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Generate output files from results"""
        try:
            return self.generate_reports(results, config)
        except Exception as e:
            logger.error(f"Error generating output files: {e}")
            return None


def main() -> Optional[Dict[str, Any]]:
    """Main entry point for CLI"""
    try:
        cli = BacktestCLI()
        config = cli.parse_arguments()

        # Setup logging level
        cli.setup_logging(config.get('log_level', 'INFO'))

        logger.info("Starting ChessTrader Backtesting Engine")
        logger.info(f"Configuration: {config}")

        # Run workflow
        results = cli.run_complete_workflow(config)

        if results:
            logger.info("Backtesting completed successfully")
            logger.info(f"Results: {list(results.keys())}")

            # Print summary
            if 'backtest_results' in results:
                performance = results['backtest_results'].get('performance_metrics', {})
                if performance:
                    print("\n=== PERFORMANCE SUMMARY ===")
                    print(f"Total Return: {performance.get('total_return', 0):.2%}")
                    print(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
                    print(f"Max Drawdown: {performance.get('max_drawdown', 0):.2%}")
                    print(f"Win Rate: {performance.get('win_rate', 0):.2%}")

            if 'report_paths' in results:
                print("\n=== OUTPUT FILES ===")
                for report_type, path in results['report_paths'].items():
                    print(f"{report_type}: {path}")

        else:
            logger.error("Backtesting failed")
            return None

        return results

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None


if __name__ == '__main__':
    main()
"""
Professional tearsheet generation with Monte Carlo uncertainty visualization

Creates comprehensive performance reports including equity curves, drawdown analysis,
monthly returns heatmaps, and statistical summaries with confidence intervals.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import seaborn, fall back to matplotlib if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

logger = logging.getLogger(__name__)


class TearsheetGenerator:
    """
    Professional tearsheet generator for backtesting results

    Creates comprehensive performance reports with Monte Carlo uncertainty
    visualization and statistical analysis integration.
    """

    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (15, 12)):
        """
        Initialize tearsheet generator

        Args:
            style: Matplotlib style for plots
            figsize: Default figure size for charts
        """
        self.style = style
        self.figsize = figsize

        # Set plotting style
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('default')
            logger.warning(f"Style '{style}' not available, using default")

        # Color scheme for consistency
        self.colors = {
            'equity': '#2E86AB',
            'drawdown': '#A23B72',
            'confidence': '#F18F01',
            'benchmark': '#C73E1D',
            'positive': '#4CAF50',
            'negative': '#F44336'
        }

    def generate_tearsheet(self, equity_curve: pd.DataFrame, performance_metrics: Dict[str, Any],
                          monte_carlo_results: Optional[pd.DataFrame] = None,
                          confidence_intervals: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate comprehensive tearsheet with all components

        Args:
            equity_curve: Time series of portfolio equity
            performance_metrics: Calculated performance statistics
            monte_carlo_results: Monte Carlo simulation results
            confidence_intervals: Confidence bounds for metrics

        Returns:
            Dictionary containing tearsheet components
        """
        logger.info("Generating comprehensive tearsheet")

        try:
            tearsheet = {}

            # Summary statistics table
            tearsheet['summary_stats'] = self._create_stats_table(
                performance_metrics, confidence_intervals
            )

            # Equity curve with uncertainty bands
            tearsheet['equity_chart'] = self._plot_equity_curve(
                equity_curve, confidence_intervals
            )

            # Drawdown analysis
            if 'equity' in equity_curve.columns:
                drawdown = self._calculate_drawdown_series(equity_curve['equity'])
                tearsheet['drawdown_chart'] = self._plot_drawdown(drawdown)

            # Monthly returns heatmap
            tearsheet['monthly_returns'] = self._create_monthly_heatmap(equity_curve)

            # Monte Carlo analysis if available
            if monte_carlo_results is not None:
                tearsheet['monte_carlo_analysis'] = self._analyze_monte_carlo(
                    monte_carlo_results, confidence_intervals
                )

            logger.info("Tearsheet generation completed successfully")
            return tearsheet

        except Exception as e:
            logger.error(f"Error generating tearsheet: {e}")
            return {'error': str(e)}

    def _create_stats_table(self, metrics: Dict[str, Any],
                           confidence_intervals: Optional[Dict] = None) -> Dict[str, Any]:
        """Create formatted statistics table with confidence intervals"""
        try:
            # Format metrics for display
            formatted_stats = {}

            # Core performance metrics
            if 'total_return' in metrics:
                formatted_stats['Total Return'] = f"{metrics['total_return']:.1%}"
            if 'annualized_return' in metrics:
                formatted_stats['Annualized Return'] = f"{metrics['annualized_return']:.1%}"
            if 'sharpe_ratio' in metrics:
                formatted_stats['Sharpe Ratio'] = f"{metrics['sharpe_ratio']:.2f}"
            if 'sortino_ratio' in metrics:
                formatted_stats['Sortino Ratio'] = f"{metrics['sortino_ratio']:.2f}"
            if 'max_drawdown' in metrics:
                formatted_stats['Max Drawdown'] = f"{metrics['max_drawdown']:.1%}"
            if 'volatility' in metrics:
                formatted_stats['Volatility'] = f"{metrics['volatility']:.1%}"
            if 'win_rate' in metrics:
                formatted_stats['Win Rate'] = f"{metrics['win_rate']:.1%}"
            if 'profit_factor' in metrics:
                formatted_stats['Profit Factor'] = f"{metrics['profit_factor']:.2f}"
            if 'calmar_ratio' in metrics:
                formatted_stats['Calmar Ratio'] = f"{metrics['calmar_ratio']:.2f}"

            # Add confidence ranges if available
            confidence_ranges = {}
            if confidence_intervals:
                for metric, intervals in confidence_intervals.items():
                    if '95%' in intervals:
                        low, high = intervals['95%']
                        confidence_ranges[metric] = f"[{low:.2%}, {high:.2%}]"

            return {
                'formatted_stats': formatted_stats,
                'confidence_ranges': confidence_ranges
            }

        except Exception as e:
            logger.error(f"Error creating stats table: {e}")
            return {'formatted_stats': {}, 'confidence_ranges': {}}

    def _plot_equity_curve(self, equity_curve: pd.DataFrame,
                          confidence_intervals: Optional[Dict] = None):
        """Plot equity curve with Monte Carlo confidence bands"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            if equity_curve.empty:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
                return fig

            # Plot main equity curve
            if 'equity' in equity_curve.columns:
                dates = equity_curve.index
                equity = equity_curve['equity']

                ax.plot(dates, equity, color=self.colors['equity'], linewidth=2, label='Portfolio Value')

                # Add confidence bands if available
                if confidence_intervals and 'total_return' in confidence_intervals:
                    initial_value = equity.iloc[0]
                    ci_95 = confidence_intervals['total_return'].get('95%', (None, None))

                    if ci_95[0] is not None and ci_95[1] is not None:
                        lower_bound = initial_value * (1 + ci_95[0])
                        upper_bound = initial_value * (1 + ci_95[1])

                        ax.fill_between(dates, lower_bound, upper_bound,
                                      alpha=0.2, color=self.colors['confidence'],
                                      label='95% Confidence Interval')

            ax.set_title('Portfolio Equity Curve with Uncertainty Bands', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.xticks(rotation=45)

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error plotting equity curve: {e}")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
            return fig

    def _plot_drawdown(self, drawdown_series: pd.Series):
        """Plot drawdown underwater chart"""
        try:
            fig, ax = plt.subplots(figsize=(12, 4))

            if drawdown_series.empty:
                ax.text(0.5, 0.5, 'No drawdown data available', ha='center', va='center')
                return fig

            # Convert to percentage and plot
            drawdown_pct = drawdown_series * 100

            ax.fill_between(drawdown_series.index, drawdown_pct, 0,
                           color=self.colors['drawdown'], alpha=0.7)
            ax.plot(drawdown_series.index, drawdown_pct,
                   color=self.colors['drawdown'], linewidth=1)

            ax.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown (%)')
            ax.grid(True, alpha=0.3)

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.xticks(rotation=45)

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error plotting drawdown: {e}")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
            return fig

    def _create_monthly_heatmap(self, equity_curve: pd.DataFrame):
        """Create monthly returns heatmap"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            if equity_curve.empty or 'returns' not in equity_curve.columns:
                ax.text(0.5, 0.5, 'No returns data for heatmap', ha='center', va='center')
                return fig

            # Calculate monthly returns
            returns = equity_curve['returns'].dropna()
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

            # Create pivot table for heatmap
            monthly_returns.index = pd.to_datetime(monthly_returns.index)
            pivot_data = monthly_returns.groupby([
                monthly_returns.index.year,
                monthly_returns.index.month
            ]).first().unstack()

            # Plot heatmap
            if HAS_SEABORN:
                sns.heatmap(pivot_data * 100, annot=True, fmt='.1f', cmap='RdYlGn',
                           center=0, ax=ax, cbar_kws={'label': 'Monthly Return (%)'})
            else:
                # Fallback to matplotlib imshow
                im = ax.imshow(pivot_data * 100, cmap='RdYlGn', aspect='auto')
                ax.figure.colorbar(im, ax=ax, label='Monthly Return (%)')

            ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Year')

            return fig

        except Exception as e:
            logger.error(f"Error creating monthly heatmap: {e}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
            return fig

    def _analyze_monte_carlo(self, monte_carlo_results: pd.DataFrame,
                            confidence_intervals: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze Monte Carlo results for uncertainty quantification"""
        try:
            analysis = {}

            # Distribution plots
            analysis['distribution_plots'] = self._plot_monte_carlo_distributions(monte_carlo_results)

            # Risk metrics
            analysis['risk_metrics'] = self._calculate_risk_metrics(monte_carlo_results)

            # Uncertainty summary
            analysis['uncertainty_summary'] = self._create_uncertainty_summary(
                monte_carlo_results, confidence_intervals
            )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing Monte Carlo results: {e}")
            return {'error': str(e)}

    def _plot_monte_carlo_distributions(self, results: pd.DataFrame):
        """Plot distribution of Monte Carlo results"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()

            metrics_to_plot = ['total_return', 'sharpe_ratio', 'max_drawdown']

            for i, metric in enumerate(metrics_to_plot):
                if metric in results.columns and i < len(axes):
                    data = results[metric].dropna()

                    axes[i].hist(data, bins=50, alpha=0.7, color=self.colors['equity'])
                    axes[i].axvline(data.median(), color=self.colors['drawdown'],
                                  linestyle='--', label=f'Median: {data.median():.2f}')
                    axes[i].set_title(f'{metric.replace("_", " ").title()} Distribution')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)

            # Remove unused subplot
            if len(metrics_to_plot) < len(axes):
                fig.delaxes(axes[-1])

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error plotting Monte Carlo distributions: {e}")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
            return fig

    def _calculate_risk_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk metrics from Monte Carlo results"""
        try:
            risk_metrics = {}

            if 'total_return' in results.columns:
                returns = results['total_return'].dropna()
                risk_metrics['var_95'] = np.percentile(returns, 5)  # 95% VaR
                risk_metrics['cvar_95'] = returns[returns <= risk_metrics['var_95']].mean()  # 95% CVaR
                risk_metrics['worst_case_scenario'] = returns.min()
                risk_metrics['best_case_scenario'] = returns.max()
                risk_metrics['probability_of_loss'] = (returns < 0).mean()

            return risk_metrics

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}

    def _create_uncertainty_summary(self, results: pd.DataFrame,
                                  confidence_intervals: Optional[Dict]) -> Dict[str, str]:
        """Create text summary of uncertainty analysis"""
        try:
            summary = {}

            if 'total_return' in results.columns:
                returns = results['total_return']
                summary['return_range'] = f"Expected return range: {returns.quantile(0.05):.1%} to {returns.quantile(0.95):.1%}"
                summary['median_return'] = f"Median return: {returns.median():.1%}"

            if 'max_drawdown' in results.columns:
                dd = results['max_drawdown']
                summary['drawdown_range'] = f"Drawdown range: {dd.quantile(0.95):.1%} to {dd.quantile(0.05):.1%}"

            return summary

        except Exception as e:
            logger.error(f"Error creating uncertainty summary: {e}")
            return {}

    def _calculate_drawdown_series(self, equity_series: pd.Series) -> pd.Series:
        """Calculate drawdown time series"""
        try:
            peak = equity_series.cummax()
            drawdown = (equity_series / peak) - 1
            return drawdown
        except Exception as e:
            logger.error(f"Error calculating drawdown series: {e}")
            return pd.Series()

    def generate_full_report(self, equity_curve: pd.DataFrame, performance_metrics: Dict[str, Any],
                           monte_carlo_results: Optional[pd.DataFrame] = None,
                           confidence_intervals: Optional[Dict] = None,
                           output_path: str = None) -> str:
        """Generate complete tearsheet report and save to file"""
        try:
            # Generate all components
            tearsheet = self.generate_tearsheet(
                equity_curve, performance_metrics, monte_carlo_results, confidence_intervals
            )

            # Create output path if not provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"tearsheet_{timestamp}"

            # Save individual charts
            if 'equity_chart' in tearsheet and tearsheet['equity_chart'] is not None:
                tearsheet['equity_chart'].savefig(f"{output_path}_equity.png", dpi=300, bbox_inches='tight')

            if 'drawdown_chart' in tearsheet and tearsheet['drawdown_chart'] is not None:
                tearsheet['drawdown_chart'].savefig(f"{output_path}_drawdown.png", dpi=300, bbox_inches='tight')

            if 'monthly_returns' in tearsheet and tearsheet['monthly_returns'] is not None:
                tearsheet['monthly_returns'].savefig(f"{output_path}_monthly.png", dpi=300, bbox_inches='tight')

            logger.info(f"Full report generated: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error generating full report: {e}")
            return None
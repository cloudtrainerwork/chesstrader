"""
Test suite for tearsheet generator

Tests professional performance report generation with Monte Carlo
uncertainty visualization and comprehensive analysis.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for testing
import matplotlib.pyplot as plt

from src.backtesting.performance.tearsheet import TearsheetGenerator
from src.backtesting.performance.reporting import ReportGenerator


class TestTearsheetGenerator:
    """Test suite for tearsheet generation"""

    def setup_method(self):
        """Setup test data"""
        self.generator = TearsheetGenerator()

        # Create sample data
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        returns = np.random.normal(0.001, 0.02, 252)
        equity_values = 100000 * np.cumprod(1 + returns)

        self.equity_curve = pd.DataFrame({
            'date': dates,
            'equity': equity_values,
            'returns': returns
        })

        # Mock Monte Carlo results
        self.monte_carlo_results = pd.DataFrame({
            'total_return': np.random.normal(0.15, 0.05, 1000),
            'sharpe_ratio': np.random.normal(1.2, 0.3, 1000),
            'max_drawdown': np.random.normal(-0.12, 0.03, 1000)
        })

        # Mock confidence intervals
        self.confidence_intervals = {
            'total_return': {'90%': (0.08, 0.22), '95%': (0.05, 0.25), '99%': (0.01, 0.29)},
            'sharpe_ratio': {'90%': (0.7, 1.7), '95%': (0.6, 1.8), '99%': (0.5, 1.9)},
            'max_drawdown': {'90%': (-0.18, -0.06), '95%': (-0.21, -0.03), '99%': (-0.25, 0.01)}
        }

    def test_tearsheet_generation(self):
        """Test comprehensive tearsheet generation"""
        # Generate tearsheet with all components
        tearsheet = self.generator.generate_tearsheet(
            equity_curve=self.equity_curve,
            performance_metrics={
                'total_return': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.12,
                'win_rate': 0.65,
                'profit_factor': 1.8
            },
            monte_carlo_results=self.monte_carlo_results,
            confidence_intervals=self.confidence_intervals
        )

        # Verify tearsheet structure
        assert 'summary_stats' in tearsheet
        assert 'equity_chart' in tearsheet
        assert 'drawdown_chart' in tearsheet
        assert 'monthly_returns' in tearsheet
        assert 'monte_carlo_analysis' in tearsheet

        # Verify each component has expected content
        assert len(tearsheet['summary_stats']) > 0
        assert tearsheet['equity_chart'] is not None
        assert tearsheet['drawdown_chart'] is not None

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_equity_curve_plot(self, mock_show, mock_savefig):
        """Test equity curve plotting with confidence bands"""
        fig = self.generator._plot_equity_curve(
            self.equity_curve,
            self.confidence_intervals
        )

        # Should create a figure
        assert fig is not None
        assert hasattr(fig, 'axes')

        # Should have confidence bands for uncertainty
        ax = fig.axes[0]
        assert len(ax.collections) > 0  # Fill_between creates collections

        plt.close(fig)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_drawdown_plot(self, mock_show, mock_savefig):
        """Test drawdown visualization"""
        # Calculate drawdown series
        cummax = self.equity_curve['equity'].cummax()
        drawdown = (self.equity_curve['equity'] / cummax - 1) * 100

        fig = self.generator._plot_drawdown(drawdown)

        assert fig is not None
        assert len(fig.axes) >= 1

        # Should show negative drawdown values
        ax = fig.axes[0]
        y_data = ax.lines[0].get_ydata()
        assert np.any(y_data <= 0)  # Drawdown should be negative/zero

        plt.close(fig)

    def test_monthly_returns_heatmap(self):
        """Test monthly returns heatmap creation"""
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            result = self.generator._create_monthly_heatmap(self.equity_curve)

            # Should call plotting functions
            mock_subplots.assert_called_once()
            mock_ax.set_title.assert_called()

            assert result == mock_fig

    def test_stats_table_creation(self):
        """Test statistics table formatting"""
        metrics = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.12,
            'volatility': 0.18,
            'win_rate': 0.65,
            'profit_factor': 1.8
        }

        stats_table = self.generator._create_stats_table(
            metrics,
            self.confidence_intervals
        )

        # Verify table structure
        assert isinstance(stats_table, dict)
        assert 'formatted_stats' in stats_table
        assert 'confidence_ranges' in stats_table

        # Check key metrics are included
        formatted = stats_table['formatted_stats']
        assert 'Total Return' in str(formatted)
        assert 'Sharpe Ratio' in str(formatted)
        assert 'Max Drawdown' in str(formatted)

    def test_monte_carlo_integration(self):
        """Test Monte Carlo uncertainty visualization"""
        mc_analysis = self.generator._analyze_monte_carlo(
            self.monte_carlo_results,
            self.confidence_intervals
        )

        assert 'distribution_plots' in mc_analysis
        assert 'risk_metrics' in mc_analysis
        assert 'uncertainty_summary' in mc_analysis

        # Risk metrics should include VaR/CVaR estimates
        risk_metrics = mc_analysis['risk_metrics']
        assert 'var_95' in risk_metrics
        assert 'cvar_95' in risk_metrics
        assert 'worst_case_scenario' in risk_metrics

    @patch('matplotlib.pyplot.savefig')
    def test_full_report_generation(self, mock_savefig):
        """Test complete report generation workflow"""
        # Mock file operations
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            # Generate full report
            report_path = self.generator.generate_full_report(
                equity_curve=self.equity_curve,
                performance_metrics={'total_return': 0.15, 'sharpe_ratio': 1.2},
                monte_carlo_results=self.monte_carlo_results,
                confidence_intervals=self.confidence_intervals,
                output_path='/tmp/test_report'
            )

            # Should generate files
            assert report_path is not None
            assert mock_savefig.called

    def test_error_handling(self):
        """Test tearsheet generation with missing data"""
        # Empty equity curve
        empty_equity = pd.DataFrame()

        tearsheet = self.generator.generate_tearsheet(
            equity_curve=empty_equity,
            performance_metrics={},
            monte_carlo_results=None,
            confidence_intervals=None
        )

        # Should handle gracefully
        assert tearsheet is not None
        assert 'error' not in tearsheet or tearsheet.get('error') is None


class TestReportGenerator:
    """Test suite for report generation"""

    def setup_method(self):
        """Setup test data"""
        self.generator = ReportGenerator()

    def test_html_report_generation(self):
        """Test HTML report formatting"""
        tearsheet_data = {
            'summary_stats': {'total_return': '15.0%', 'sharpe_ratio': '1.2'},
            'equity_chart': Mock(),
            'monte_carlo_analysis': {'risk_metrics': {'var_95': -0.08}}
        }

        html_report = self.generator.generate_html_report(tearsheet_data)

        assert isinstance(html_report, str)
        assert '<html>' in html_report
        assert 'Total Return' in html_report
        assert '15.0%' in html_report

    @patch('builtins.open', create=True)
    def test_pdf_report_generation(self, mock_open):
        """Test PDF report creation"""
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        with patch('weasyprint.HTML') as mock_html:
            mock_pdf = Mock()
            mock_html.return_value.write_pdf = Mock(return_value=mock_pdf)

            result = self.generator.generate_pdf_report(
                html_content="<html><body>Test</body></html>",
                output_path="/tmp/test.pdf"
            )

            assert result is not None
            mock_open.assert_called()

    def test_csv_export(self):
        """Test CSV data export"""
        metrics_data = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.12
        }

        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            self.generator.export_metrics_csv(metrics_data, '/tmp/metrics.csv')
            mock_to_csv.assert_called_once_with('/tmp/metrics.csv', index=False)
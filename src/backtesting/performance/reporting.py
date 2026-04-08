"""
Report generation and formatting for tearsheet output

Handles HTML, PDF, and CSV export of performance analysis results
with professional formatting and styling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import json
import logging
from datetime import datetime
import base64
from io import BytesIO

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Professional report generator for backtesting results

    Formats tearsheet data into various output formats including HTML, PDF,
    and CSV for distribution and analysis.
    """

    def __init__(self):
        """Initialize report generator"""
        self.html_template = self._get_html_template()

    def generate_html_report(self, tearsheet_data: Dict[str, Any]) -> str:
        """
        Generate HTML report from tearsheet data

        Args:
            tearsheet_data: Dictionary containing tearsheet components

        Returns:
            Formatted HTML string
        """
        try:
            logger.info("Generating HTML report")

            # Extract data components
            stats = tearsheet_data.get('summary_stats', {})
            mc_analysis = tearsheet_data.get('monte_carlo_analysis', {})

            # Format statistics table
            stats_html = self._format_stats_table(stats)

            # Format Monte Carlo analysis
            mc_html = self._format_monte_carlo_analysis(mc_analysis)

            # Convert charts to base64 for embedding
            charts_html = self._embed_charts(tearsheet_data)

            # Generate final HTML
            html_content = self.html_template.format(
                title="Portfolio Performance Report",
                generated_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                summary_stats=stats_html,
                monte_carlo_analysis=mc_html,
                charts=charts_html
            )

            logger.info("HTML report generated successfully")
            return html_content

        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return f"<html><body><h1>Error generating report: {e}</h1></body></html>"

    def generate_pdf_report(self, html_content: str, output_path: str) -> Optional[str]:
        """
        Generate PDF report from HTML content

        Args:
            html_content: Formatted HTML content
            output_path: Path to save PDF file

        Returns:
            Path to generated PDF file or None if failed
        """
        try:
            # Try to use weasyprint for HTML to PDF conversion
            try:
                import weasyprint

                html_doc = weasyprint.HTML(string=html_content)
                html_doc.write_pdf(output_path)

                logger.info(f"PDF report generated: {output_path}")
                return output_path

            except ImportError:
                logger.warning("weasyprint not available, using matplotlib for basic PDF")
                return self._generate_matplotlib_pdf(output_path)

        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return None

    def export_metrics_csv(self, metrics_data: Dict[str, Any], output_path: str) -> bool:
        """
        Export performance metrics to CSV format

        Args:
            metrics_data: Performance metrics dictionary
            output_path: Path to save CSV file

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Exporting metrics to CSV: {output_path}")

            # Convert metrics to DataFrame
            df = pd.DataFrame([metrics_data])

            # Save to CSV
            df.to_csv(output_path, index=False)

            logger.info("CSV export completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error exporting metrics to CSV: {e}")
            return False

    def export_monte_carlo_results(self, mc_results: pd.DataFrame, output_path: str) -> bool:
        """
        Export Monte Carlo simulation results to CSV

        Args:
            mc_results: Monte Carlo results DataFrame
            output_path: Path to save CSV file

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Exporting Monte Carlo results to CSV: {output_path}")

            # Add summary statistics
            summary_stats = mc_results.describe()

            # Save both detailed results and summary
            with pd.ExcelWriter(output_path.replace('.csv', '.xlsx')) as writer:
                mc_results.to_excel(writer, sheet_name='Simulation_Results', index=False)
                summary_stats.to_excel(writer, sheet_name='Summary_Statistics')

            # Also save as CSV for simple access
            mc_results.to_csv(output_path, index=False)

            logger.info("Monte Carlo export completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error exporting Monte Carlo results: {e}")
            return False

    def _format_stats_table(self, stats: Dict[str, Any]) -> str:
        """Format statistics table as HTML"""
        try:
            if not stats or 'formatted_stats' not in stats:
                return "<p>No statistics available</p>"

            formatted_stats = stats['formatted_stats']
            confidence_ranges = stats.get('confidence_ranges', {})

            html = '<table class="stats-table">\n'
            html += '<tr><th>Metric</th><th>Value</th><th>95% Confidence Interval</th></tr>\n'

            for metric, value in formatted_stats.items():
                confidence = confidence_ranges.get(metric.lower().replace(' ', '_'), 'N/A')
                html += f'<tr><td>{metric}</td><td>{value}</td><td>{confidence}</td></tr>\n'

            html += '</table>'
            return html

        except Exception as e:
            logger.error(f"Error formatting stats table: {e}")
            return f"<p>Error formatting statistics: {e}</p>"

    def _format_monte_carlo_analysis(self, mc_analysis: Dict[str, Any]) -> str:
        """Format Monte Carlo analysis as HTML"""
        try:
            if not mc_analysis:
                return "<p>No Monte Carlo analysis available</p>"

            html = '<div class="monte-carlo-section">\n'
            html += '<h2>Monte Carlo Risk Analysis</h2>\n'

            # Risk metrics
            risk_metrics = mc_analysis.get('risk_metrics', {})
            if risk_metrics:
                html += '<h3>Risk Metrics</h3>\n'
                html += '<ul>\n'
                for metric, value in risk_metrics.items():
                    formatted_value = f"{value:.2%}" if isinstance(value, (int, float)) else str(value)
                    metric_name = metric.replace('_', ' ').title()
                    html += f'<li><strong>{metric_name}:</strong> {formatted_value}</li>\n'
                html += '</ul>\n'

            # Uncertainty summary
            uncertainty = mc_analysis.get('uncertainty_summary', {})
            if uncertainty:
                html += '<h3>Uncertainty Analysis</h3>\n'
                html += '<ul>\n'
                for key, summary in uncertainty.items():
                    html += f'<li>{summary}</li>\n'
                html += '</ul>\n'

            html += '</div>'
            return html

        except Exception as e:
            logger.error(f"Error formatting Monte Carlo analysis: {e}")
            return f"<p>Error formatting Monte Carlo analysis: {e}</p>"

    def _embed_charts(self, tearsheet_data: Dict[str, Any]) -> str:
        """Convert matplotlib figures to base64 for HTML embedding"""
        try:
            charts_html = ""

            chart_types = [
                ('equity_chart', 'Portfolio Equity Curve'),
                ('drawdown_chart', 'Drawdown Analysis'),
                ('monthly_returns', 'Monthly Returns Heatmap')
            ]

            for chart_key, chart_title in chart_types:
                if chart_key in tearsheet_data and tearsheet_data[chart_key] is not None:
                    try:
                        fig = tearsheet_data[chart_key]

                        # Convert to base64
                        buffer = BytesIO()
                        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                        buffer.seek(0)
                        image_base64 = base64.b64encode(buffer.getvalue()).decode()
                        buffer.close()

                        # Add to HTML
                        charts_html += f'<div class="chart-section">\n'
                        charts_html += f'<h3>{chart_title}</h3>\n'
                        charts_html += f'<img src="data:image/png;base64,{image_base64}" alt="{chart_title}" style="width: 100%; max-width: 800px;">\n'
                        charts_html += '</div>\n'

                    except Exception as e:
                        logger.warning(f"Could not embed chart {chart_key}: {e}")

            return charts_html

        except Exception as e:
            logger.error(f"Error embedding charts: {e}")
            return "<p>Error embedding charts</p>"

    def _get_html_template(self) -> str:
        """Return HTML template for report generation"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .stats-table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }}
        .stats-table th, .stats-table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        .stats-table th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .chart-section {{
            margin: 30px 0;
            text-align: center;
        }}
        .monte-carlo-section {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>Generated on: {generated_date}</p>
    </div>

    <div class="summary-section">
        <h2>Performance Summary</h2>
        {summary_stats}
    </div>

    <div class="charts-section">
        <h2>Performance Charts</h2>
        {charts}
    </div>

    {monte_carlo_analysis}

    <div class="footer">
        <p>Generated by ChessTrader Backtesting Engine</p>
    </div>
</body>
</html>
        """

    def _generate_matplotlib_pdf(self, output_path: str) -> Optional[str]:
        """Generate basic PDF using matplotlib when weasyprint not available"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages

            with PdfPages(output_path) as pdf:
                # Create simple text-based report
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.text(0.5, 0.5, 'Performance Report\n\n(Install weasyprint for full HTML->PDF conversion)',
                       ha='center', va='center', fontsize=14)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                pdf.savefig(fig)
                plt.close(fig)

            return output_path

        except Exception as e:
            logger.error(f"Error generating matplotlib PDF: {e}")
            return None
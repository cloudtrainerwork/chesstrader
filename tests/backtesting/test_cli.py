"""
Test suite for CLI interface

Tests command-line interface for complete backtesting workflow
orchestration including configuration and output management.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.backtesting.cli.backtest_runner import BacktestCLI


class TestBacktestCLI:
    """Test suite for CLI interface"""

    def setup_method(self):
        """Setup test fixtures"""
        self.cli = BacktestCLI()

        # Mock configuration
        self.test_config = {
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'strategy': 'iron_condor',
            'symbol': 'SPY',
            'n_simulations': 100,
            'output_dir': '/tmp/backtest_output'
        }

    def test_cli_workflow(self):
        """Test complete CLI workflow orchestration"""
        # Mock all the required components
        with patch('src.backtesting.cli.backtest_runner.BacktestEngine') as mock_engine, \
             patch('src.backtesting.cli.backtest_runner.WalkForwardOptimizer') as mock_optimizer, \
             patch('src.backtesting.cli.backtest_runner.MonteCarloSimulator') as mock_simulator, \
             patch('src.backtesting.cli.backtest_runner.TearsheetGenerator') as mock_tearsheet:

            # Setup mocks
            mock_engine_instance = Mock()
            mock_engine.return_value = mock_engine_instance
            mock_engine_instance.run_backtest.return_value = {
                'portfolio': Mock(),
                'trades': pd.DataFrame(),
                'performance': {'total_return': 0.15}
            }

            # Test the complete workflow
            result = self.cli.run_complete_workflow(self.test_config)

            # Verify workflow execution
            assert result is not None
            assert 'backtest_results' in result
            assert 'tearsheet_path' in result

            # Verify components were called
            mock_engine.assert_called_once()
            mock_engine_instance.run_backtest.assert_called_once()

    def test_argument_parsing(self):
        """Test command-line argument parsing"""
        test_args = [
            '--start-date', '2023-01-01',
            '--end-date', '2023-12-31',
            '--strategy', 'iron_condor',
            '--symbol', 'SPY',
            '--output-dir', '/tmp/test'
        ]

        # Mock sys.argv
        with patch('sys.argv', ['backtest_runner.py'] + test_args):
            config = self.cli.parse_arguments()

            assert config['start_date'] == '2023-01-01'
            assert config['end_date'] == '2023-12-31'
            assert config['strategy'] == 'iron_condor'
            assert config['symbol'] == 'SPY'
            assert config['output_dir'] == '/tmp/test'

    def test_backtest_execution(self):
        """Test basic backtest execution"""
        with patch('src.backtesting.cli.backtest_runner.BacktestEngine') as mock_engine:
            mock_engine_instance = Mock()
            mock_engine.return_value = mock_engine_instance

            # Mock return data
            mock_portfolio = Mock()
            mock_portfolio.get_equity_curve.return_value = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=252, freq='D'),
                'equity': np.random.randn(252).cumsum() + 100000
            })
            mock_portfolio.get_performance_metrics.return_value = {
                'total_return': 0.15,
                'sharpe_ratio': 1.2
            }

            mock_engine_instance.run_backtest.return_value = {
                'portfolio': mock_portfolio,
                'trades': pd.DataFrame(),
                'performance': {'total_return': 0.15}
            }

            result = self.cli.run_backtest(self.test_config)

            assert result is not None
            assert 'portfolio' in result
            mock_engine_instance.run_backtest.assert_called_once()

    def test_walk_forward_optimization(self):
        """Test walk-forward optimization execution"""
        with patch('src.backtesting.cli.backtest_runner.WalkForwardOptimizer') as mock_optimizer:
            mock_optimizer_instance = Mock()
            mock_optimizer.return_value = mock_optimizer_instance

            mock_optimizer_instance.optimize.return_value = pd.DataFrame({
                'period_start': ['2023-01-01', '2023-07-01'],
                'period_end': ['2023-06-30', '2023-12-31'],
                'best_params': [{'param1': 1}, {'param1': 2}],
                'return': [0.12, 0.18],
                'sharpe': [1.1, 1.3]
            })

            result = self.cli.run_walk_forward(self.test_config)

            assert result is not None
            mock_optimizer_instance.optimize.assert_called_once()

    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation execution"""
        with patch('src.backtesting.cli.backtest_runner.MonteCarloSimulator') as mock_simulator:
            mock_simulator_instance = Mock()
            mock_simulator.return_value = mock_simulator_instance

            # Mock simulation results
            mock_results = pd.DataFrame({
                'total_return': np.random.normal(0.15, 0.05, 100),
                'sharpe_ratio': np.random.normal(1.2, 0.3, 100),
                'max_drawdown': np.random.normal(-0.12, 0.03, 100)
            })

            mock_simulator_instance.simulate_backtests.return_value = mock_results

            result = self.cli.run_monte_carlo(self.test_config, Mock())

            assert result is not None
            assert len(result) == 100  # Should return simulation results
            mock_simulator_instance.simulate_backtests.assert_called_once()

    def test_configuration_validation(self):
        """Test configuration validation"""
        # Valid configuration
        valid_config = {
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'strategy': 'iron_condor',
            'symbol': 'SPY'
        }

        assert self.cli.validate_configuration(valid_config) == True

        # Invalid date range
        invalid_config = {
            'start_date': '2023-12-31',
            'end_date': '2023-01-01',  # End before start
            'strategy': 'iron_condor',
            'symbol': 'SPY'
        }

        assert self.cli.validate_configuration(invalid_config) == False

        # Missing required fields
        incomplete_config = {
            'start_date': '2023-01-01',
            # Missing end_date, strategy, symbol
        }

        assert self.cli.validate_configuration(incomplete_config) == False

    def test_output_file_generation(self):
        """Test output file generation and management"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.test_config.copy()
            config['output_dir'] = temp_dir

            # Mock results
            mock_results = {
                'performance': {'total_return': 0.15, 'sharpe_ratio': 1.2},
                'equity_curve': pd.DataFrame({
                    'date': pd.date_range('2023-01-01', periods=10, freq='D'),
                    'equity': np.random.randn(10).cumsum() + 100000
                })
            }

            output_files = self.cli.generate_output_files(mock_results, config)

            assert output_files is not None
            assert 'metrics_csv' in output_files
            # Files should be generated in temp directory

    def test_progress_reporting(self):
        """Test progress reporting functionality"""
        progress_callback = Mock()

        with patch('src.backtesting.cli.backtest_runner.BacktestEngine') as mock_engine:
            mock_engine_instance = Mock()
            mock_engine.return_value = mock_engine_instance

            # Configure mock to simulate progress updates
            def side_effect(*args, **kwargs):
                progress_callback("Starting backtest...")
                progress_callback("Processing data...")
                progress_callback("Backtest complete")
                return {'portfolio': Mock(), 'performance': {}}

            mock_engine_instance.run_backtest.side_effect = side_effect

            self.cli.run_backtest(self.test_config, progress_callback=progress_callback)

            # Verify progress callbacks were made
            assert progress_callback.call_count >= 3

    def test_error_handling(self):
        """Test error handling in CLI workflow"""
        # Test with invalid strategy
        invalid_config = self.test_config.copy()
        invalid_config['strategy'] = 'invalid_strategy'

        with patch('src.backtesting.cli.backtest_runner.BacktestEngine') as mock_engine:
            mock_engine.side_effect = ValueError("Invalid strategy")

            result = self.cli.run_backtest(invalid_config)
            assert result is None  # Should handle error gracefully

        # Test with missing data
        config_no_data = self.test_config.copy()
        config_no_data['symbol'] = 'NONEXISTENT'

        with patch('src.backtesting.cli.backtest_runner.BacktestEngine') as mock_engine:
            mock_engine_instance = Mock()
            mock_engine.return_value = mock_engine_instance
            mock_engine_instance.run_backtest.side_effect = RuntimeError("No data available")

            result = self.cli.run_backtest(config_no_data)
            assert result is None  # Should handle error gracefully

    def test_main_entry_point(self):
        """Test main entry point function"""
        test_args = [
            'backtest_runner.py',
            '--start-date', '2023-01-01',
            '--end-date', '2023-12-31',
            '--strategy', 'iron_condor',
            '--symbol', 'SPY'
        ]

        with patch('sys.argv', test_args), \
             patch('src.backtesting.cli.backtest_runner.BacktestEngine'), \
             patch.object(self.cli, 'run_complete_workflow') as mock_workflow:

            mock_workflow.return_value = {'status': 'success'}

            # Test main entry point
            from src.backtesting.cli.backtest_runner import main
            result = main()

            # Should execute successfully
            assert mock_workflow.called

    def test_csv_output_generation(self):
        """Test CSV output file generation"""
        performance_data = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.12,
            'win_rate': 0.65
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            self.cli.save_metrics_to_csv(performance_data, temp_path)

            # Verify file was created and contains expected data
            assert os.path.exists(temp_path)

            df = pd.read_csv(temp_path)
            assert 'total_return' in df.columns
            assert df.iloc[0]['total_return'] == 0.15

        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
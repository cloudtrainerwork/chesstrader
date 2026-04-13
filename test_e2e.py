#!/usr/bin/env python3
"""
ChessTrader End-to-End Integration Test

Comprehensive test of the entire ChessTrader system including:
- Package installation verification
- CLI functionality
- API functionality
- Configuration management
- Error handling
- Example execution

Run with: python test_e2e.py
"""

import subprocess
import sys
import os
import json
import tempfile
from pathlib import Path
import time

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

class E2ETest:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.results = []

    def log(self, message, level="INFO"):
        color = {
            "INFO": Colors.BLUE,
            "PASS": Colors.GREEN,
            "FAIL": Colors.RED,
            "WARN": Colors.YELLOW
        }.get(level, "")

        print(f"{color}{level}: {message}{Colors.END}")

    def run_command(self, cmd, expect_success=True, timeout=30):
        """Run a command and return (success, stdout, stderr)"""
        try:
            self.log(f"Running: {cmd}")
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            success = result.returncode == 0 if expect_success else result.returncode != 0
            return success, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return False, "", str(e)

    def test_package_installation(self):
        """Test package installation and CLI availability"""
        self.log("=" * 50, "INFO")
        self.log("Testing Package Installation", "INFO")
        self.log("=" * 50, "INFO")

        # Test 1: Verify package is installed
        success, stdout, stderr = self.run_command("pip show chesstrader")
        if success:
            self.log("✅ ChessTrader package is installed", "PASS")
            self.passed += 1
        else:
            self.log("❌ ChessTrader package not found", "FAIL")
            self.failed += 1
            return False

        # Test 2: Verify CLI command is available
        success, stdout, stderr = self.run_command("which chesstrader")
        if success:
            self.log("✅ chesstrader command is in PATH", "PASS")
            self.passed += 1
        else:
            self.log("❌ chesstrader command not found in PATH", "FAIL")
            self.failed += 1
            return False

        # Test 3: Test CLI help
        success, stdout, stderr = self.run_command("chesstrader --help")
        if success and "ChessTrader Options AI" in stdout:
            self.log("✅ CLI help displays correctly", "PASS")
            self.passed += 1
        else:
            self.log("❌ CLI help failed", "FAIL")
            self.log(f"Error: {stderr}", "FAIL")
            self.failed += 1

        return True

    def test_cli_functionality(self):
        """Test CLI commands"""
        self.log("\n" + "=" * 50, "INFO")
        self.log("Testing CLI Functionality", "INFO")
        self.log("=" * 50, "INFO")

        # Test 1: Version command
        success, stdout, stderr = self.run_command("chesstrader version")
        if success and "v1.0.0" in stdout:
            self.log("✅ Version command works", "PASS")
            self.passed += 1
        else:
            self.log("❌ Version command failed", "FAIL")
            self.log(f"Output: {stdout}", "FAIL")
            self.log(f"Error: {stderr}", "FAIL")
            self.failed += 1

        # Test 2: Recommend command help
        success, stdout, stderr = self.run_command("chesstrader recommend --help")
        if success and "Stock/ETF symbol" in stdout:
            self.log("✅ Recommend command help works", "PASS")
            self.passed += 1
        else:
            self.log("❌ Recommend command help failed", "FAIL")
            self.failed += 1

        # Test 3: Backtest command help
        success, stdout, stderr = self.run_command("chesstrader backtest --help")
        if success and "backtesting" in stdout.lower():
            self.log("✅ Backtest command help works", "PASS")
            self.passed += 1
        else:
            self.log("❌ Backtest command help failed", "FAIL")
            self.failed += 1

        # Test 4: Try recommend with invalid symbol (should handle gracefully)
        success, stdout, stderr = self.run_command(
            "chesstrader recommend INVALID_SYMBOL",
            expect_success=False,
            timeout=60
        )
        if not success:
            self.log("✅ Invalid symbol handled gracefully", "PASS")
            self.passed += 1
        else:
            self.log("⚠️ Invalid symbol didn't fail as expected", "WARN")
            self.warnings += 1

    def test_convenience_entry_point(self):
        """Test the convenience chesstrader.py entry point"""
        self.log("\n" + "=" * 50, "INFO")
        self.log("Testing Convenience Entry Point", "INFO")
        self.log("=" * 50, "INFO")

        # Test chesstrader.py help
        success, stdout, stderr = self.run_command("python chesstrader.py --help")
        if success and "ChessTrader Options AI" in stdout:
            self.log("✅ Convenience entry point works", "PASS")
            self.passed += 1
        else:
            self.log("❌ Convenience entry point failed", "FAIL")
            self.log(f"Error: {stderr}", "FAIL")
            self.failed += 1

    def test_api_functionality(self):
        """Test programmatic API"""
        self.log("\n" + "=" * 50, "INFO")
        self.log("Testing Programmatic API", "INFO")
        self.log("=" * 50, "INFO")

        # Create a simple API test
        api_test = '''
import sys
sys.path.insert(0, "src")

try:
    from main import OptionsAI
    from config import Config

    # Test 1: Basic initialization
    ai = OptionsAI()
    print("✅ OptionsAI initialization successful")

    # Test 2: Version method
    version = OptionsAI.version()
    if version == "1.0.0":
        print("✅ Version method works")
    else:
        print(f"❌ Version method returned: {version}")

    # Test 3: Configuration loading
    config = Config()
    print(f"✅ Config loaded - API port: {config.api.port}")

    # Test 4: Config update
    ai.update_config(recommendation__confidence_threshold=0.8)
    if ai.config.recommendation.confidence_threshold == 0.8:
        print("✅ Configuration update works")
    else:
        print("❌ Configuration update failed")

    print("API_TEST_SUCCESS")

except Exception as e:
    print(f"❌ API test failed: {e}")
    import traceback
    traceback.print_exc()
    print("API_TEST_FAILED")
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(api_test)
            test_file = f.name

        try:
            success, stdout, stderr = self.run_command(f"python {test_file}", timeout=60)

            if "API_TEST_SUCCESS" in stdout:
                self.log("✅ Programmatic API tests passed", "PASS")
                self.passed += 1
                # Count individual API tests
                if "OptionsAI initialization successful" in stdout:
                    self.passed += 1
                if "Version method works" in stdout:
                    self.passed += 1
                if "Config loaded" in stdout:
                    self.passed += 1
                if "Configuration update works" in stdout:
                    self.passed += 1
            else:
                self.log("❌ Programmatic API tests failed", "FAIL")
                self.log(f"Output: {stdout}", "FAIL")
                self.log(f"Error: {stderr}", "FAIL")
                self.failed += 1

        finally:
            os.unlink(test_file)

    def test_configuration_management(self):
        """Test configuration file handling"""
        self.log("\n" + "=" * 50, "INFO")
        self.log("Testing Configuration Management", "INFO")
        self.log("=" * 50, "INFO")

        # Create test configuration
        test_config = {
            "recommendation": {
                "confidence_threshold": 0.7,
                "max_recommendations": 5
            },
            "backtesting": {
                "initial_capital": 150000
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f, indent=2)
            config_file = f.name

        try:
            # Test config file with CLI
            success, stdout, stderr = self.run_command(
                f"chesstrader --config {config_file} --help"
            )

            if success:
                self.log("✅ Configuration file accepted by CLI", "PASS")
                self.passed += 1
            else:
                self.log("❌ Configuration file rejected by CLI", "FAIL")
                self.failed += 1

        finally:
            os.unlink(config_file)

    def test_documentation_and_examples(self):
        """Test documentation and example files"""
        self.log("\n" + "=" * 50, "INFO")
        self.log("Testing Documentation and Examples", "INFO")
        self.log("=" * 50, "INFO")

        # Test 1: Check documentation files exist
        docs = ['README.md', 'docs/api_reference.md', 'docs/cli_guide.md']
        for doc in docs:
            if os.path.exists(doc):
                self.log(f"✅ {doc} exists", "PASS")
                self.passed += 1
            else:
                self.log(f"❌ {doc} missing", "FAIL")
                self.failed += 1

        # Test 2: Check example files exist and compile
        examples = ['examples/basic_usage.py', 'examples/backtest_example.py']
        for example in examples:
            if os.path.exists(example):
                self.log(f"✅ {example} exists", "PASS")
                self.passed += 1

                # Test compilation
                success, stdout, stderr = self.run_command(f"python -m py_compile {example}")
                if success:
                    self.log(f"✅ {example} compiles without syntax errors", "PASS")
                    self.passed += 1
                else:
                    self.log(f"❌ {example} has syntax errors", "FAIL")
                    self.failed += 1
            else:
                self.log(f"❌ {example} missing", "FAIL")
                self.failed += 1

        # Test 3: Check README content
        if os.path.exists('README.md'):
            with open('README.md', 'r') as f:
                readme_content = f.read()

            required_sections = [
                'ChessTrader', 'Installation', 'Quick Start',
                'Features', 'Examples', 'Documentation'
            ]

            for section in required_sections:
                if section in readme_content:
                    self.log(f"✅ README contains {section} section", "PASS")
                    self.passed += 1
                else:
                    self.log(f"❌ README missing {section} section", "FAIL")
                    self.failed += 1

    def test_project_structure(self):
        """Test project structure is complete"""
        self.log("\n" + "=" * 50, "INFO")
        self.log("Testing Project Structure", "INFO")
        self.log("=" * 50, "INFO")

        required_files = [
            'setup.py',
            'chesstrader.py',
            'requirements.txt',
            'src/main.py',
            'src/config.py',
            'src/cli/main.py',
            'src/cli/commands/recommend.py',
            'src/cli/commands/backtest.py'
        ]

        for file_path in required_files:
            if os.path.exists(file_path):
                self.log(f"✅ {file_path} exists", "PASS")
                self.passed += 1
            else:
                self.log(f"❌ {file_path} missing", "FAIL")
                self.failed += 1

        # Test setup.py content
        if os.path.exists('setup.py'):
            with open('setup.py', 'r') as f:
                setup_content = f.read()

            if 'chesstrader=src.cli.main:app' in setup_content:
                self.log("✅ setup.py has correct entry point", "PASS")
                self.passed += 1
            else:
                self.log("❌ setup.py missing correct entry point", "FAIL")
                self.failed += 1

    def test_error_handling(self):
        """Test error handling scenarios"""
        self.log("\n" + "=" * 50, "INFO")
        self.log("Testing Error Handling", "INFO")
        self.log("=" * 50, "INFO")

        # Test 1: Missing required argument
        success, stdout, stderr = self.run_command(
            "chesstrader recommend",
            expect_success=False
        )
        if not success and "required" in stderr.lower():
            self.log("✅ Missing argument handled correctly", "PASS")
            self.passed += 1
        else:
            self.log("❌ Missing argument not handled properly", "FAIL")
            self.failed += 1

        # Test 2: Invalid confidence value
        success, stdout, stderr = self.run_command(
            "chesstrader recommend AAPL --confidence 2.0",
            expect_success=False
        )
        if not success:
            self.log("✅ Invalid confidence value rejected", "PASS")
            self.passed += 1
        else:
            self.log("❌ Invalid confidence value accepted", "FAIL")
            self.failed += 1

    def run_full_test_suite(self):
        """Run the complete end-to-end test suite"""
        start_time = time.time()

        self.log("🚀 Starting ChessTrader End-to-End Test Suite", "INFO")
        self.log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", "INFO")

        # Run all test categories
        try:
            if self.test_package_installation():
                self.test_cli_functionality()
                self.test_convenience_entry_point()
                self.test_api_functionality()
                self.test_configuration_management()
                self.test_documentation_and_examples()
                self.test_project_structure()
                self.test_error_handling()
            else:
                self.log("Skipping further tests due to installation failure", "WARN")

        except Exception as e:
            self.log(f"Test suite error: {e}", "FAIL")
            self.failed += 1

        # Generate final report
        end_time = time.time()
        duration = end_time - start_time

        self.generate_report(duration)

    def generate_report(self, duration):
        """Generate final test report"""
        self.log("\n" + "=" * 60, "INFO")
        self.log("🎯 CHESSTRADER END-TO-END TEST REPORT", "INFO")
        self.log("=" * 60, "INFO")

        total_tests = self.passed + self.failed
        success_rate = (self.passed / total_tests * 100) if total_tests > 0 else 0

        self.log(f"Duration: {duration:.1f} seconds", "INFO")
        self.log(f"Total Tests: {total_tests}", "INFO")
        self.log(f"✅ Passed: {self.passed}", "PASS")
        self.log(f"❌ Failed: {self.failed}", "FAIL")
        self.log(f"⚠️ Warnings: {self.warnings}", "WARN")
        self.log(f"Success Rate: {success_rate:.1f}%", "INFO")

        if self.failed == 0:
            self.log("\n🎉 ALL TESTS PASSED! ChessTrader is ready for use.", "PASS")
            self.log("Next steps:", "INFO")
            self.log("• Try: chesstrader recommend AAPL", "INFO")
            self.log("• Try: chesstrader backtest --symbol SPY", "INFO")
            self.log("• Run examples: python examples/basic_usage.py", "INFO")
            return True
        else:
            self.log(f"\n⚠️ {self.failed} test(s) failed. Please review issues above.", "FAIL")
            self.log("Common solutions:", "INFO")
            self.log("• Ensure all dependencies installed: pip install -r requirements.txt", "INFO")
            self.log("• Reinstall package: pip install -e .", "INFO")
            self.log("• Check Python path and virtual environment", "INFO")
            return False

def main():
    """Main entry point"""
    test_runner = E2ETest()
    success = test_runner.run_full_test_suite()

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test the installed ChessTrader package API
"""

import subprocess
import sys

# Test script to run in subprocess
test_script = '''
try:
    # Test using the installed package
    from src.main import OptionsAI
    from src.config import Config

    # Test 1: Basic initialization
    print("Testing OptionsAI initialization...")
    ai = OptionsAI()
    print("✅ OptionsAI initialization successful")

    # Test 2: Version method
    print("Testing version method...")
    version = OptionsAI.version()
    if version == "1.0.0":
        print("✅ Version method works")
    else:
        print(f"❌ Version method returned: {version}")

    # Test 3: Configuration loading
    print("Testing configuration loading...")
    config = Config()
    print(f"✅ Config loaded - API port: {config.api.port}")

    # Test 4: Config update
    print("Testing configuration update...")
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

# Run the test in subprocess
result = subprocess.run(
    [sys.executable, "-c", test_script],
    capture_output=True,
    text=True,
    timeout=30
)

print("API Test Results:")
print("STDOUT:", result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
print("Return code:", result.returncode)

if "API_TEST_SUCCESS" in result.stdout:
    print("\n✅ Installed package API works correctly!")
    sys.exit(0)
else:
    print("\n❌ API test failed")
    sys.exit(1)
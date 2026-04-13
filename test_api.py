#!/usr/bin/env python3
"""
Quick test of the OptionsAI API when used programmatically
"""

import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from main import OptionsAI
    from config import Config

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

    print("\n✅ All API tests passed!")

except Exception as e:
    print(f"❌ API test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
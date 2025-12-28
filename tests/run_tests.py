#!/usr/bin/env python3
# tests/run_tests.py
"""
Script to run all tests
"""

import subprocess
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_tests():
    """Run all tests using pytest."""
    print("=" * 60)
    print("RUNNING BIOMETRIC AUTHENTICATION SYSTEM TESTS")
    print("=" * 60)
    
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run pytest
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_dir, "-v", "--tb=short"],
        cwd=os.path.dirname(test_dir)
    )
    
    return result.returncode


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)

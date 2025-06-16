#!/usr/bin/env python3
"""
Simple script to run a test file with the correct import paths.
"""
import os
import sys
from pathlib import Path

# Ensure project root is in the Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Import pytest
try:
    import pytest
except ImportError:
    print("Error: pytest is not installed")
    sys.exit(1)

# Import numpy
try:
    import numpy
except ImportError:
    print("Error: numpy is not installed")
    sys.exit(1)

# Get test file from command line
if len(sys.argv) < 2:
    print("Usage: python run_simple_test.py <test_file_path>")
    sys.exit(1)

test_file = sys.argv[1]
if not os.path.exists(test_file):
    print(f"Error: Test file {test_file} does not exist")
    sys.exit(1)

print(f"Running test file: {test_file}")
print(f"Python path: {sys.path}")

# Run the test
exit_code = pytest.main(["-vxs", test_file])
sys.exit(exit_code)
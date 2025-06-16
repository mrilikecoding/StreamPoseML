#!/usr/bin/env python3
"""
Script to run a specific test file with proper import path setup.
"""

import os
import sys
import importlib
from pathlib import Path

# Add the project root to the sys.path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Get the test file from the command line
if len(sys.argv) < 2:
    print("Usage: python run_module_test.py <test_file>")
    print("Example: python run_module_test.py stream_pose_ml/tests/geometry/test_distance.py")
    sys.exit(1)

test_file = sys.argv[1]

# Verify the test file exists
if not os.path.exists(test_file):
    print(f"Error: Test file '{test_file}' does not exist")
    sys.exit(1)

print(f"Running test: {test_file}")
print(f"Python path: {sys.path}")
print()

# Import pytest
try:
    import pytest
except ImportError:
    print("Error: pytest is not installed. Please install it with:")
    print("  pip install pytest")
    sys.exit(1)

# Enable pytest's assertion rewriting
import _pytest.assertion.rewrite
_pytest.assertion.rewrite.enable_assertion_pass_hook()

# Create a clean module name from the file path
module_path = os.path.splitext(test_file)[0]
module_name = module_path.replace('/', '.')

print(f"Importing module: {module_name}")

# Try to import the test module
try:
    # Remove the file extension
    spec = importlib.util.spec_from_file_location(module_name, test_file)
    if spec is None:
        print(f"Error: Could not create spec for module '{module_name}' from file '{test_file}'")
        sys.exit(1)
    
    test_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = test_module
    spec.loader.exec_module(test_module)
    
    print(f"Successfully imported: {module_name}")
    print("Module contents:")
    for name in dir(test_module):
        if name.startswith('Test'):
            print(f"  {name}")
    
    # Run pytest on the test file
    print("\nRunning pytest...\n")
    result = pytest.main(["-v", test_file])
    sys.exit(result)
except Exception as e:
    print(f"Error importing module: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
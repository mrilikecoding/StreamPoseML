#!/usr/bin/env python3
"""
Script to run tests for the stream_pose_ml module after project restructuring.
"""

import os
import sys
from pathlib import Path
import importlib.util
import importlib

# Add the project root to the Python path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Path to the module tests
TESTS_PATH = os.path.join(project_root, "stream_pose_ml", "tests")


def check_module_structure():
    """Check if the module structure is correct."""
    print("Checking module structure...")
    
    # Verify core modules exist
    modules_to_check = [
        "stream_pose_ml.geometry.distance",
        "stream_pose_ml.geometry.vector",
        "stream_pose_ml.geometry.joint",
        "stream_pose_ml.geometry.angle",
        "stream_pose_ml.blaze_pose.blaze_pose_frame",
        "stream_pose_ml.blaze_pose.blaze_pose_sequence",
    ]
    
    for module_name in modules_to_check:
        try:
            # Try to import the module
            module = importlib.import_module(module_name)
            print(f"  ✓ {module_name} is importable")
        except ImportError as e:
            print(f"  ✗ {module_name} cannot be imported: {e}")
            parts = module_name.split('.')
            if len(parts) >= 3:
                # Check if the module is available at a different location
                alt_module = f"stream_pose_ml.{parts[-1]}"
                try:
                    alt = importlib.import_module(alt_module)
                    print(f"    However, {alt_module} is importable")
                except ImportError:
                    pass
    print()


def fix_test_imports():
    """Fix relative imports in test files."""
    print("Fixing any remaining import issues in test files...")
    
    for root, _, files in os.walk(TESTS_PATH):
        for file in files:
            if file.endswith('.py') and not file == '__init__.py':
                file_path = os.path.join(root, file)
                
                # Read the file content
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check for import references that need to be fixed
                if "from stream_pose_ml.geometry." in content:
                    rel_module = os.path.relpath(root, TESTS_PATH)
                    print(f"  Checking imports in {rel_module}/{file}")
                    
                    # Check if the file needs to use sys.path to fix imports
                    if "sys.path.insert" not in content:
                        print(f"    File needs sys.path fix")
                        
                        # Add sys.path fix at the top of the file
                        path_setup = """
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[3]  # /Users/nathangreen/Development/stream_pose_ml
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
"""
                        # Add the path setup to the file
                        with open(file_path, 'w') as f:
                            f.write(path_setup + content)
    print()


def run_tests():
    """Run the module tests."""
    # First check the module structure
    check_module_structure()
    
    # Fix any remaining import issues
    fix_test_imports()
    
    import pytest
    
    # Run the tests
    print(f"Running tests in {TESTS_PATH}")
    result = pytest.main([
        "-xvs",  # Show verbose output, stop on first failure
        TESTS_PATH  # Path to the tests
    ])
    
    return result


if __name__ == "__main__":
    sys.exit(run_tests())
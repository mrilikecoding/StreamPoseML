#!/usr/bin/env python3
"""
Script to run all tests in the correct order.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def run_command(command, cwd=None):
    """Run a shell command and return its output and exit code."""
    print(f"Running: {' '.join(command)}")
    process = subprocess.run(
        command,
        cwd=cwd or project_root,
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True
    )
    print(process.stdout)
    return process.returncode

def install_dependencies():
    """Install required dependencies for tests."""
    print("\n=== Installing Dependencies ===\n")
    try:
        run_command(["pip", "install", "-e", "."])
        run_command(["pip", "install", "pytest", "numpy"])
        return True
    except Exception as e:
        print(f"Error installing dependencies: {e}")
        return False

def run_all_tests():
    """Run both API and module tests."""
    # First run the API tests
    print("\n=== Running API Tests ===\n")
    api_result = run_command(["python", "-m", "pytest", "tests", "-v"])
    
    # Then run the module tests
    print("\n=== Running Module Tests ===\n")
    module_result = run_command(["python", "-m", "pytest", "stream_pose_ml/tests", "-v"])
    
    return api_result == 0 and module_result == 0

def main():
    """Main function to run the tests."""
    # First make sure all dependencies are installed
    if not install_dependencies():
        print("Failed to install dependencies")
        return 1
    
    # Run all tests
    print("\nRunning all tests...")
    success = run_all_tests()
    
    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
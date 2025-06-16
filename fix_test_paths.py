#!/usr/bin/env python3
"""
Script to fix import paths in test files for easier test running.
"""

import os
import re
import sys
from pathlib import Path

# Test files directory
TESTS_PATH = "/Users/nathangreen/Development/stream_pose_ml/stream_pose_ml/tests"

def fix_test_file(file_path):
    """Fix import paths in a test file."""
    print(f"Processing: {os.path.relpath(file_path, '/Users/nathangreen/Development/stream_pose_ml')}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if "import sys" in content and "project_root" in content:
        print("  Already fixed, skipping")
        return False
    
    # Add sys.path manipulation to the beginning of the file
    path_setup = """import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[3]  # /Users/nathangreen/Development/stream_pose_ml
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

"""
    
    # Find the first import statement
    import_match = re.search(r'^(import|from)\s+', content, re.MULTILINE)
    
    if import_match:
        # Insert before first import
        pos = import_match.start()
        new_content = content[:pos] + path_setup + content[pos:]
    else:
        # Add to beginning if no imports found
        new_content = path_setup + content
    
    # Write back to file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print("  Updated import paths")
    return True

def main():
    """Find and fix all test files."""
    updated_count = 0
    
    for root, _, files in os.walk(TESTS_PATH):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                file_path = os.path.join(root, file)
                if fix_test_file(file_path):
                    updated_count += 1
    
    print(f"\nUpdated {updated_count} test files")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script to fix imports in the module files.
"""

import os
import re
import sys
from pathlib import Path

# Module files directory
MODULE_DIRS = [
    "/Users/nathangreen/Development/stream_pose_ml/stream_pose_ml/blaze_pose",
    "/Users/nathangreen/Development/stream_pose_ml/stream_pose_ml/geometry",
    "/Users/nathangreen/Development/stream_pose_ml/stream_pose_ml/jobs",
    "/Users/nathangreen/Development/stream_pose_ml/stream_pose_ml/learning",
    "/Users/nathangreen/Development/stream_pose_ml/stream_pose_ml/serializers",
    "/Users/nathangreen/Development/stream_pose_ml/stream_pose_ml/services",
    "/Users/nathangreen/Development/stream_pose_ml/stream_pose_ml/transformers",
    "/Users/nathangreen/Development/stream_pose_ml/stream_pose_ml/utils",
]

def fix_imports_in_file(file_path):
    """Fix imports in a module file."""
    print(f"Processing: {os.path.relpath(file_path, '/Users/nathangreen/Development/stream_pose_ml')}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix imports that reference parent directory modules
    # E.g., from ..geometry.joint import Joint -> from stream_pose_ml.geometry.joint import Joint
    pattern = r'from \.\.([\w]+)\.([\w]+) import ([\w, ]+)'
    new_content = re.sub(pattern, r'from stream_pose_ml.\1.\2 import \3', content)
    
    # Fix imports within the same module directory 
    # E.g., from .joint import Joint -> from stream_pose_ml.geometry.joint import Joint
    module_path = os.path.dirname(file_path)
    module_name = os.path.basename(module_path)
    pattern = r'from \.([\w]+) import ([\w, ]+)'
    new_content = re.sub(pattern, rf'from stream_pose_ml.{module_name}.\1 import \2', new_content)
    
    # Only write if changes were made
    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        print(f"  Updated imports")
        return True
    else:
        print("  No changes needed")
        return False

def main():
    """Find and fix all module files."""
    updated_count = 0
    
    for module_dir in MODULE_DIRS:
        if not os.path.exists(module_dir):
            print(f"Warning: Directory {module_dir} does not exist")
            continue
            
        for root, _, files in os.walk(module_dir):
            for file in files:
                if file.endswith('.py') and file != '__init__.py':
                    file_path = os.path.join(root, file)
                    if fix_imports_in_file(file_path):
                        updated_count += 1
    
    print(f"\nUpdated {updated_count} module files")

if __name__ == "__main__":
    main()
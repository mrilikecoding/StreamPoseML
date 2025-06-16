#!/usr/bin/env python3
"""
Script to fix imports in test files after project restructuring.
"""

import os
import re
import sys
from pathlib import Path


def fix_imports_in_file(file_path):
    """Fix imports in a test file."""
    print(f"Fixing imports in {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if we've already fixed this file by looking for the sys.path addition
    if "sys.path.insert(0, str(project_root))" in content:
        print(f"  Already fixed, skipping")
        return
    
    # Find the import statements
    import_pattern = re.compile(r'from \.([a-zA-Z0-9_]+) import (.+)')
    imports = import_pattern.findall(content)
    
    # Prepare the new imports section
    path_setup = """import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[3]  # /Users/nathangreen/Development/stream_pose_ml
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
"""
    
    # Replace the relative imports with absolute imports
    new_content = content
    for module, names in imports:
        old_import = f"from .{module} import {names}"
        
        # Get the module parent from the file path
        rel_path = os.path.relpath(file_path, "/Users/nathangreen/Development/stream_pose_ml/stream_pose_ml/tests")
        module_parent = os.path.dirname(rel_path).replace(os.path.sep, '.')
        
        # Convert relative imports to absolute imports based on the file location
        if 'geometry' in module_parent:
            new_import = f"from stream_pose_ml.geometry.{module} import {names}"
        elif 'blaze_pose' in module_parent:
            new_import = f"from stream_pose_ml.blaze_pose.{module} import {names}"
        elif 'jobs' in module_parent:
            new_import = f"from stream_pose_ml.jobs.{module} import {names}"
        elif 'learning' in module_parent:
            new_import = f"from stream_pose_ml.learning.{module} import {names}"
        elif 'services' in module_parent:
            new_import = f"from stream_pose_ml.services.{module} import {names}"
        elif 'serializers' in module_parent:
            new_import = f"from stream_pose_ml.serializers.{module} import {names}"
        elif 'transformers' in module_parent:
            new_import = f"from stream_pose_ml.transformers.{module} import {names}"
        elif 'utils' in module_parent:
            new_import = f"from stream_pose_ml.utils.{module} import {names}"
        else:
            # Default case - for imports from the same directory
            if module_parent:
                new_import = f"from stream_pose_ml.{module_parent}.{module} import {names}"
            else:
                new_import = f"from stream_pose_ml.{module} import {names}"
        
        new_content = new_content.replace(old_import, new_import)
    
    # Add the path setup after the existing imports but before the class definitions
    import_section_end = re.search(r'import.*\n\n', new_content, re.DOTALL)
    if import_section_end:
        pos = import_section_end.end()
        new_content = new_content[:pos] + path_setup + new_content[pos:]
    else:
        # If we couldn't find the end of the import section, add it after the docstring
        docstring_end = re.search(r'""".+?"""\n', new_content, re.DOTALL)
        if docstring_end:
            pos = docstring_end.end()
            new_content = new_content[:pos] + path_setup + new_content[pos:]
        else:
            # Last resort: add it at the beginning of the file
            new_content = path_setup + new_content
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"  Fixed imports in {file_path}")


def main():
    """Fix imports in all test files."""
    tests_dir = "/Users/nathangreen/Development/stream_pose_ml/stream_pose_ml/tests"
    
    for root, _, files in os.walk(tests_dir):
        for file in files:
            if file.endswith('.py') and not file == '__init__.py':
                file_path = os.path.join(root, file)
                fix_imports_in_file(file_path)


if __name__ == "__main__":
    main()
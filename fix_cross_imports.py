#!/usr/bin/env python
"""
A script to fix cross-module imports in the stream_pose_ml module.
This script converts:
    from .xxx.yyy import zzz
to 
    from ..xxx.yyy import zzz

when accessing a sibling module.
"""

import os
import re
from pathlib import Path

def fix_cross_imports(directory):
    """Fix imports across subdirectories"""
    print(f"Fixing cross-module imports in {directory}")
    
    # Get all subdirectories
    modules = []
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)) and not item.startswith("__"):
            modules.append(item)
    
    print(f"Found modules: {modules}")
    
    for module in modules:
        module_dir = os.path.join(directory, module)
        
        for root, _, files in os.walk(module_dir):
            for file in files:
                if not file.endswith('.py') or file == "__init__.py":
                    continue
                    
                file_path = os.path.join(root, file)
                
                with open(file_path, 'r') as f:
                    content = f.read()
                
                updated_content = content
                changes_made = False
                
                # For each other module, check if we're importing from it
                for other_module in modules:
                    if other_module == module:
                        continue
                        
                    pattern = re.compile(f'from \\.{other_module}\\.(\\w+) import ([^\\n]+)')
                    if pattern.search(content):
                        updated_content = pattern.sub(f'from ..{other_module}.\\1 import \\2', updated_content)
                        changes_made = True
                
                if changes_made:
                    print(f"  Fixing imports in {file_path}")
                    with open(file_path, 'w') as f:
                        f.write(updated_content)

if __name__ == "__main__":
    # Update imports in the stream_pose_ml module
    base_dir = Path(__file__).parent
    module_dir = base_dir / "stream_pose_ml"
    
    fix_cross_imports(module_dir)
    print("Done!")
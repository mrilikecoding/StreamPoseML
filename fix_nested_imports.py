#!/usr/bin/env python
"""
A script to fix nested imports in the stream_pose_ml module.
This script converts:
    from .xxx.yyy import zzz
to 
    from .yyy import zzz

when inside the xxx directory.
"""

import os
import re
from pathlib import Path

def fix_nested_imports(directory):
    """Fix imports in the given directory."""
    print(f"Fixing nested imports in {directory}")
    
    for root, _, files in os.walk(directory):
        dir_name = os.path.basename(root)
        if dir_name == "__pycache__":
            continue
            
        parent_pattern = re.compile(f'from \\.{dir_name}\\.(\\w+) import ([^\\n]+)')
        
        for file in files:
            if not file.endswith('.py'):
                continue
                
            file_path = os.path.join(root, file)
            # print(f"  Checking {file_path}")
            
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Replace the imports
            updated_content = parent_pattern.sub(r'from .\1 import \2', content)
            
            if content != updated_content:
                print(f"  Fixing imports in {file_path}")
                with open(file_path, 'w') as f:
                    f.write(updated_content)

if __name__ == "__main__":
    # Update imports in the stream_pose_ml module
    base_dir = Path(__file__).parent
    module_dir = base_dir / "stream_pose_ml"
    
    fix_nested_imports(module_dir)
    print("Done!")
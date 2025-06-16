#!/usr/bin/env python
"""
A script to update imports in the stream_pose_ml module.
This script converts:
    from stream_pose_ml.xxx import yyy
to 
    from .xxx import yyy
"""

import os
import re
from pathlib import Path

def fix_imports(directory):
    """Fix imports in the given directory."""
    print(f"Fixing imports in {directory}")
    pattern = re.compile(r'from\s+stream_pose_ml\.([^\s]+)\s+import\s+([^\n]+)')
    
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith('.py') or file == '__init__.py':
                continue
                
            file_path = os.path.join(root, file)
            print(f"  Processing {file_path}")
            
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Replace the imports
            updated_content = pattern.sub(r'from .\1 import \2', content)
            
            if content != updated_content:
                with open(file_path, 'w') as f:
                    f.write(updated_content)
                print(f"    Updated imports in {file_path}")

if __name__ == "__main__":
    # Update imports in the stream_pose_ml module
    base_dir = Path(__file__).parent
    module_dir = base_dir / "stream_pose_ml"
    
    fix_imports(module_dir)
    print("Done!")
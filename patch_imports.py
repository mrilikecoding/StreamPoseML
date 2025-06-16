#!/usr/bin/env python3
"""
A script to patch import paths in the module files.
"""

import os
import sys
import re
from pathlib import Path

def patch_file(file_path):
    """Replace imports in a file."""
    print(f"Patching {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace 'from stream_pose_ml.xxx' with 'from .xxx'
    patched = re.sub(r'from stream_pose_ml\.([a-zA-Z0-9_\.]+) import', r'from .\1 import', content)
    
    # Only write if changed
    if patched != content:
        with open(file_path, 'w') as f:
            f.write(patched)
        print(f"  Updated imports")
        return True
    else:
        print(f"  No changes needed")
        return False

# Patch blaze_pose_frame.py
patch_file('stream_pose_ml/blaze_pose/blaze_pose_frame.py')


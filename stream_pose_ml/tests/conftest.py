"""
Conftest file for module tests.
"""
import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parents[2]  # /Users/nathangreen/Development/stream_pose_ml
module_root = Path(__file__).parents[1]   # /Users/nathangreen/Development/stream_pose_ml/stream_pose_ml

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    
# Print debugging info
print(f"Project root: {project_root}")
print(f"Module root: {module_root}")
print(f"Python path: {sys.path}")

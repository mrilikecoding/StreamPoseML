"""
Pytest configuration file for stream_pose_ml tests.
"""
import sys
import os
from pathlib import Path

# Add parent to Python path if needed
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
"""
Root conftest.py to configure pytest for both API and module tests.
"""
import sys
import os
from pathlib import Path

# Add the project root to sys.path for proper import resolution
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Print Python path for debugging
print(f"Python path in root conftest.py: {sys.path}")
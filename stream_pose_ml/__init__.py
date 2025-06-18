"""
Stream Pose ML module.
"""

# Add all top-level modules for easier importing
import sys
import os
from pathlib import Path


# Make top-level components accessible directly from the module
from stream_pose_ml.stream_pose_client import StreamPoseClient
from .ml_flow_client import MLFlowClient

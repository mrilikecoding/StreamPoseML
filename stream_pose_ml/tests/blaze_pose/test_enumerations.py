"""Tests for the enumerations in the blaze_pose module."""

import sys
from enum import Enum
from pathlib import Path

from stream_pose_ml.blaze_pose.enumerations import BlazePoseJoints, blaze_pose_joints

# Add the project root to the Python path
project_root = Path(__file__).parents[
    3
]  # /Users/nathangreen/Development/stream_pose_ml
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class TestBlazePoseJoints:
    """Tests for the BlazePoseJoints enum."""

    def test_blaze_pose_joints_contains_expected_joint_names(self):
        """
        GIVEN the blaze_pose_joints tuple
        WHEN examining its contents
        THEN it contains the expected joint names
        """
        # Check that the tuple contains the essential joints
        essential_joints = [
            "nose",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]

        for joint in essential_joints:
            assert joint in blaze_pose_joints

    def test_blaze_pose_joints_enum_created_properly(self):
        """
        GIVEN the BlazePoseJoints enum
        WHEN examining its structure
        THEN it is properly created from the blaze_pose_joints tuple
        """
        # Verify it's an Enum
        assert issubclass(BlazePoseJoints, Enum)

        # Verify all joints from the tuple are in the enum
        for i, joint_name in enumerate(blaze_pose_joints):
            # Enum values are 1-indexed by default
            assert BlazePoseJoints[joint_name].value == i + 1

        # Verify the total number of joints
        assert len(BlazePoseJoints) == len(blaze_pose_joints)

    def test_blaze_pose_joints_enum_usage(self):
        """
        GIVEN the BlazePoseJoints enum
        WHEN using it to reference joints
        THEN it works as expected for enum access
        """
        # Access by name
        assert BlazePoseJoints.nose.name == "nose"
        assert BlazePoseJoints.left_shoulder.name == "left_shoulder"

        # Convert to list of names
        joint_names = [joint.name for joint in BlazePoseJoints]
        assert isinstance(joint_names, list)
        assert "nose" in joint_names
        assert "left_shoulder" in joint_names
        assert "right_hip" in joint_names

        # Iterate through enum
        for joint in BlazePoseJoints:
            assert isinstance(joint, BlazePoseJoints)
            assert joint.name in blaze_pose_joints

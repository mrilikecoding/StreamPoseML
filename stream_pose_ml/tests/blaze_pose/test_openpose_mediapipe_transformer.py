"""Tests for the OpenPoseMediapipeTransformer class."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parents[
    3
]  # /Users/nathangreen/Development/stream_pose_ml
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ruff: noqa: E402
from stream_pose_ml.blaze_pose.blaze_pose_frame import BlazePoseFrame
from stream_pose_ml.blaze_pose.openpose_mediapipe_transformer import (
    OpenPoseMediapipeTransformer,
    OpenPoseMediapipeTransformerError,
)
from stream_pose_ml.geometry.joint import Joint
from stream_pose_ml.geometry.vector import Vector


class TestOpenPoseMediapipeTransformer:
    """Tests for the OpenPoseMediapipeTransformer class."""

    def test_open_pose_distance_definition_map(self):
        """
        GIVEN the OpenPoseMediapipeTransformer class
        WHEN open_pose_distance_definition_map is called
        THEN a dictionary of distance definitions is returned
        """
        # Act
        distance_map = OpenPoseMediapipeTransformer.open_pose_distance_definition_map()

        # Assert
        assert isinstance(distance_map, dict)
        assert len(distance_map) > 0

        # Check format of entries
        for name, definition in distance_map.items():
            assert isinstance(name, str)
            assert isinstance(definition, tuple)
            assert len(definition) == 2
            assert all(isinstance(item, str) for item in definition)

        # Check some specific entries
        assert "nose_to_plumb_line" in distance_map
        assert distance_map["nose_to_plumb_line"] == ("nose", "plumb_line")
        assert "right_shoulder_to_plumb_line" in distance_map
        assert distance_map["right_shoulder_to_plumb_line"] == (
            "right_shoulder",
            "plumb_line",
        )

    def test_open_pose_angle_definition_map(self):
        """
        GIVEN the OpenPoseMediapipeTransformer class
        WHEN open_pose_angle_definition_map is called
        THEN a dictionary of angle definitions is returned
        """
        # Act
        angle_map = OpenPoseMediapipeTransformer.open_pose_angle_definition_map()

        # Assert
        assert isinstance(angle_map, dict)
        assert len(angle_map) > 0

        # Check format of entries
        for name, definition in angle_map.items():
            assert isinstance(name, str)
            assert isinstance(definition, tuple)
            assert len(definition) == 2
            assert all(isinstance(item, str) for item in definition)

        # Check some specific entries
        assert "nose_neck_to_plumb_line" in angle_map
        assert angle_map["nose_neck_to_plumb_line"] == ("nose_neck", "plumb_line")
        assert "neck_right_shoulder_to_right_shoulder_right_elbow" in angle_map
        assert angle_map["neck_right_shoulder_to_right_shoulder_right_elbow"] == (
            "neck_right_shoulder",
            "right_shoulder_right_elbow",
        )

    def test_create_openpose_joints_and_vectors_success(self):
        """
        GIVEN a BlazePoseFrame with joint positions
        WHEN create_openpose_joints_and_vectors is called
        THEN OpenPose joints and vectors are added to the frame
        """
        # Arrange
        mock_frame = MagicMock(spec=BlazePoseFrame)
        mock_frame.has_joint_positions = True
        mock_frame.joints = {
            "left_shoulder": MagicMock(spec=Joint),
            "right_shoulder": MagicMock(spec=Joint),
            "left_hip": MagicMock(spec=Joint),
            "right_hip": MagicMock(spec=Joint),
            "nose": MagicMock(spec=Joint),
            "left_eye": MagicMock(spec=Joint),
            "right_eye": MagicMock(spec=Joint),
            "left_ear": MagicMock(spec=Joint),
            "right_ear": MagicMock(spec=Joint),
            "left_elbow": MagicMock(spec=Joint),
            "right_elbow": MagicMock(spec=Joint),
            "left_wrist": MagicMock(spec=Joint),
            "right_wrist": MagicMock(spec=Joint),
            "left_knee": MagicMock(spec=Joint),
            "right_knee": MagicMock(spec=Joint),
            "left_ankle": MagicMock(spec=Joint),
            "right_ankle": MagicMock(spec=Joint),
            "left_heel": MagicMock(spec=Joint),
            "right_heel": MagicMock(spec=Joint),
            "left_foot_index": MagicMock(spec=Joint),
            "right_foot_index": MagicMock(spec=Joint),
        }
        mock_frame.vectors = {}

        # Mock get_average_joint and get_vector methods
        mock_frame.get_average_joint.return_value = MagicMock(spec=Joint)
        mock_frame.get_vector.return_value = MagicMock(spec=Vector)

        # Act
        result = OpenPoseMediapipeTransformer.create_openpose_joints_and_vectors(
            mock_frame
        )

        # Assert
        assert result is True

        # Check that the required joints are created
        assert mock_frame.get_average_joint.call_count == 2
        mock_frame.get_average_joint.assert_any_call(
            name="neck", joint_1="left_shoulder", joint_2="right_shoulder"
        )
        mock_frame.get_average_joint.assert_any_call(
            name="mid_hip", joint_1="left_hip", joint_2="right_hip"
        )

        # Check that the required vectors are created
        assert mock_frame.get_vector.call_count > 20  # Many vectors are created
        mock_frame.get_vector.assert_any_call("plumb_line", "neck", "mid_hip")
        mock_frame.get_vector.assert_any_call("nose_neck", "nose", "neck")

    def test_create_openpose_joints_and_vectors_no_joint_positions(self):
        """
        GIVEN a BlazePoseFrame without joint positions
        WHEN create_openpose_joints_and_vectors is called
        THEN False is returned and no calculations are performed
        """
        # Arrange
        mock_frame = MagicMock(spec=BlazePoseFrame)
        mock_frame.has_joint_positions = False

        # Act
        result = OpenPoseMediapipeTransformer.create_openpose_joints_and_vectors(
            mock_frame
        )

        # Assert
        assert result is False
        assert not mock_frame.get_average_joint.called
        assert not mock_frame.get_vector.called

    def test_create_openpose_joints_and_vectors_error(self):
        """
        GIVEN a BlazePoseFrame that will cause an error during processing
        WHEN create_openpose_joints_and_vectors is called
        THEN OpenPoseMediapipeTransformerError is raised
        """
        # Arrange
        mock_frame = MagicMock(spec=BlazePoseFrame)
        mock_frame.has_joint_positions = True

        # Mock method to raise an exception
        mock_frame.get_average_joint.side_effect = Exception("Test error")

        # Act & Assert
        with pytest.raises(
            OpenPoseMediapipeTransformerError, match="Problem setting joints or vectors"
        ):
            OpenPoseMediapipeTransformer.create_openpose_joints_and_vectors(mock_frame)


class TestOpenPoseMediapipeTransformerIntegration:
    """Integration tests for OpenPoseMediapipeTransformer with mocked BlazePoseFrame."""

    def test_integration_create_openpose_joints_and_vectors(self):
        """
        GIVEN a mocked BlazePoseFrame with required joint data
        WHEN create_openpose_joints_and_vectors is called
        THEN the frame is updated with the expected joints and vectors
        """
        # Create a completely mocked frame instead of trying to use a real one
        mock_frame = MagicMock()
        mock_frame.has_joint_positions = True
        mock_frame.joints = {}
        mock_frame.vectors = {}

        # Mock the get_average_joint and get_vector methods
        mock_average_joint = MagicMock(spec=Joint)
        mock_vector = MagicMock(spec=Vector)
        mock_frame.get_average_joint.return_value = mock_average_joint
        mock_frame.get_vector.return_value = mock_vector

        # Act
        result = OpenPoseMediapipeTransformer.create_openpose_joints_and_vectors(
            mock_frame
        )

        # Assert
        assert result is True

        # Verify that the methods were called to create joints and vectors
        assert mock_frame.get_average_joint.call_count >= 2
        assert mock_frame.get_vector.call_count >= 20

        # Check that the joints and vectors dictionaries were updated
        assert len(mock_frame.joints) > 0
        assert len(mock_frame.vectors) > 0

        # Verify that get_average_joint was called with the expected arguments
        mock_frame.get_average_joint.assert_any_call(
            name="neck", joint_1="left_shoulder", joint_2="right_shoulder"
        )
        mock_frame.get_average_joint.assert_any_call(
            name="mid_hip", joint_1="left_hip", joint_2="right_hip"
        )

        # Verify that get_vector was called with expected arguments
        mock_frame.get_vector.assert_any_call("plumb_line", "neck", "mid_hip")
        mock_frame.get_vector.assert_any_call("nose_neck", "nose", "neck")

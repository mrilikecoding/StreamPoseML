import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[3]  # Adjust if needed
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from unittest.mock import MagicMock, patch

import pytest

from stream_pose_ml.blaze_pose.blaze_pose_frame import BlazePoseFrame
from stream_pose_ml.serializers.blaze_pose_frame_serializer import (
    BlazePoseFrameSerializer,
)


class TestBlazePoseFrameSerializer:
    """Test the BlazePoseFrameSerializer class."""

    @pytest.fixture
    def mock_joint_serializer(self):
        """Create a mock for JointSerializer."""
        with patch(
            "stream_pose_ml.serializers.blaze_pose_frame_serializer.JointSerializer"
        ) as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            mock_instance.serialize.return_value = {
                "type": "Joint",
                "name": "test_joint",
            }
            yield mock

    @pytest.fixture
    def mock_angle_serializer(self):
        """Create a mock for AngleSerializer."""
        with patch(
            "stream_pose_ml.serializers.blaze_pose_frame_serializer.AngleSerializer"
        ) as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            mock_instance.serialize.return_value = {
                "type": "Angle",
                "name": "test_angle",
            }
            yield mock

    @pytest.fixture
    def mock_distance_serializer(self):
        """Create a mock for DistanceSerializer."""
        with patch(
            "stream_pose_ml.serializers.blaze_pose_frame_serializer.DistanceSerializer"
        ) as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            mock_instance.serialize.return_value = {
                "type": "Distance",
                "name": "test_distance",
            }
            yield mock

    @pytest.fixture
    def blaze_pose_frame(self):
        """Create a mock BlazePoseFrame object for testing."""
        frame = MagicMock(spec=BlazePoseFrame)
        frame.sequence_id = "seq123"
        frame.sequence_source = "video1.mp4"
        frame.frame_number = 42
        frame.image_dimensions = {"width": 640, "height": 480}
        frame.has_joint_positions = True

        # Create mock joints, angles, and distances
        joint1 = MagicMock()
        joint2 = MagicMock()
        angle1 = MagicMock()
        distance1 = MagicMock()

        frame.joints = {"joint1": joint1, "joint2": joint2}
        frame.angles = {"angle1": angle1}
        frame.distances = {"distance1": distance1}

        return frame

    def test_serialize_with_geometry(
        self,
        blaze_pose_frame,
        mock_joint_serializer,
        mock_angle_serializer,
        mock_distance_serializer,
    ):
        """Test the serialize method with geometry data."""
        # Given
        serializer = BlazePoseFrameSerializer()

        # When
        result = serializer.serialize(blaze_pose_frame)

        # Then
        assert result["type"] == "BlasePoseFrame"
        assert result["sequence_id"] == "seq123"
        assert result["sequence_source"] == "video1.mp4"
        assert result["frame_number"] == 42
        assert result["image_dimensions"] == {"width": 640, "height": 480}
        assert result["has_joint_positions"] is True

        # Check if serializers were called
        assert len(result["joint_positions"]) == 2
        assert "joint1" in result["joint_positions"]
        assert "joint2" in result["joint_positions"]
        assert len(result["angles"]) == 1
        assert "angle1" in result["angles"]
        assert len(result["distances"]) == 1
        assert "distance1" in result["distances"]

        # Verify serializer calls
        mock_joint_serializer.return_value.serialize.assert_any_call(
            blaze_pose_frame.joints["joint1"]
        )
        mock_joint_serializer.return_value.serialize.assert_any_call(
            blaze_pose_frame.joints["joint2"]
        )
        mock_angle_serializer.return_value.serialize.assert_called_once_with(
            blaze_pose_frame.angles["angle1"]
        )
        mock_distance_serializer.return_value.serialize.assert_called_once_with(
            blaze_pose_frame.distances["distance1"]
        )

    def test_serialize_without_joint_positions(
        self,
        blaze_pose_frame,
        mock_joint_serializer,
        mock_angle_serializer,
        mock_distance_serializer,
    ):
        """Test the serialize method without joint positions."""
        # Given
        blaze_pose_frame.has_joint_positions = False
        serializer = BlazePoseFrameSerializer()

        # When
        result = serializer.serialize(blaze_pose_frame)

        # Then
        assert result["has_joint_positions"] is False
        assert result["joint_positions"] == {}
        assert result["angles"] == {}
        assert result["distances"] == {}

        # Verify serializers were not called
        mock_joint_serializer.return_value.serialize.assert_not_called()
        mock_angle_serializer.return_value.serialize.assert_not_called()
        mock_distance_serializer.return_value.serialize.assert_not_called()

    def test_serialize_static_method(
        self,
        blaze_pose_frame,
        mock_joint_serializer,
        mock_angle_serializer,
        mock_distance_serializer,
    ):
        """Test the serialize method as a static method."""
        # When
        result = BlazePoseFrameSerializer.serialize(blaze_pose_frame)

        # Then
        assert result["type"] == "BlasePoseFrame"
        assert result["sequence_id"] == "seq123"
        assert result["sequence_source"] == "video1.mp4"
        assert result["frame_number"] == 42

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[3]  # Adjust if needed
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from unittest.mock import patch

import pytest

from stream_pose_ml.serializers.labeled_frame_serializer import LabeledFrameSerializer


class TestLabeledFrameSerializer:
    """Test the LabeledFrameSerializer class."""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame for testing."""
        return {
            "video_id": "test_video",
            "weight_transfer_type": "forward",
            "step_type": "stride",
            "data": {
                "frame_number": 42,
                "joint_positions": {
                    "nose": {
                        "x": 0.5,
                        "y": 0.6,
                        "z": 0.7,
                        "x_normalized": 320,
                        "y_normalized": 288,
                        "z_normalized": 350,
                    },
                    "shoulder": {
                        "x": 0.4,
                        "y": 0.5,
                        "z": 0.6,
                        "x_normalized": 256,
                        "y_normalized": 240,
                        "z_normalized": 300,
                    },
                },
                "angles": {
                    "elbow": {"angle_2d_degrees": 45.5, "angle_3d_degrees": 60.2},
                    "knee": {"angle_2d_degrees": 90.1, "angle_3d_degrees": 85.7},
                },
                "distances": {
                    "hand_to_hip": {
                        "distance_2d": 50.5,
                        "distance_3d": 75.2,
                        "distance_2d_normalized": 0.5,
                        "distance_3d_normalized": 0.75,
                    }
                },
            },
        }

    def test_init(self):
        """Test initialization with different parameters."""
        # Default initialization
        serializer = LabeledFrameSerializer()
        assert serializer.include_angles is True
        assert serializer.include_distances is True
        assert serializer.include_joints is False
        assert serializer.include_normalized is True
        assert serializer.include_z_axis is False

        # Custom initialization
        serializer = LabeledFrameSerializer(
            include_angles=False,
            include_distances=False,
            include_joints=True,
            include_normalized=False,
            include_z_axis=True,
        )
        assert serializer.include_angles is False
        assert serializer.include_distances is False
        assert serializer.include_joints is True
        assert serializer.include_normalized is False
        assert serializer.include_z_axis is True

    def test_serialize_basic_info(self, sample_frame):
        """Test serialization of basic frame information."""
        # Given
        serializer = LabeledFrameSerializer(
            include_angles=False, include_distances=False, include_joints=False
        )

        # When
        result = serializer.serialize(sample_frame)

        # Then
        assert result["video_id"] == "test_video"
        assert result["frame_number"] == 42
        assert result["weight_transfer_type"] == "forward"
        assert result["step_type"] == "stride"
        assert (
            result["step_frame_id"] == 0
        )  # Since both weight_transfer_type and step_type are not None

    def test_serialize_null_values(self):
        """Test serialization with null label values."""
        # Given
        frame = {
            "video_id": "test_video",
            "weight_transfer_type": None,
            "step_type": None,
            "data": {
                "frame_number": 42,
                "angles": {},
                "distances": {},
                "joint_positions": {},
            },
        }

        serializer = LabeledFrameSerializer()

        # When
        result = serializer.serialize(frame)

        # Then
        assert result["weight_transfer_type"] == "NULL"
        assert result["step_type"] == "NULL"
        assert result["step_frame_id"] == "NULL"

    def test_serialize_angles(self, sample_frame):
        """Test serialization of angles."""
        # Given
        serializer = LabeledFrameSerializer(
            include_angles=True, include_distances=False, include_joints=False
        )

        # When
        with patch.object(LabeledFrameSerializer, "serialize_angles") as mock_serialize:
            mock_serialize.return_value = {"test": "angles"}
            result = serializer.serialize(sample_frame)

        # Then
        assert "angles" in result
        assert result["angles"] == {"test": "angles"}
        mock_serialize.assert_called_once_with(sample_frame["data"]["angles"])

    def test_serialize_distances(self, sample_frame):
        """Test serialization of distances."""
        # Given
        serializer = LabeledFrameSerializer(
            include_angles=False,
            include_distances=True,
            include_joints=False,
            include_normalized=True,
            include_z_axis=True,
        )

        # When
        with patch.object(
            LabeledFrameSerializer, "serialize_distances"
        ) as mock_serialize:
            mock_serialize.return_value = {"test": "distances"}
            result = serializer.serialize(sample_frame)

        # Then
        assert "distances" in result
        assert result["distances"] == {"test": "distances"}
        mock_serialize.assert_called_once_with(
            distances=sample_frame["data"]["distances"],
            include_normalized=True,
            include_z_axis=True,
        )

    def test_serialize_joints(self, sample_frame):
        """Test serialization of joints."""
        # Given
        serializer = LabeledFrameSerializer(
            include_angles=False,
            include_distances=False,
            include_joints=True,
            include_normalized=True,
            include_z_axis=True,
        )

        # When
        with patch.object(LabeledFrameSerializer, "serialize_joints") as mock_serialize:
            mock_serialize.return_value = {"test": "joints"}
            result = serializer.serialize(sample_frame)

        # Then
        assert "joints" in result
        assert result["joints"] == {"test": "joints"}
        mock_serialize.assert_called_once_with(
            sample_frame["data"]["joint_positions"],
            include_normalized=True,
            include_z_axis=True,
        )

    def test_serialize_angles_static(self):
        """Test the static serialize_angles method."""
        # Given
        angles = {
            "elbow": {"angle_2d_degrees": 45.5, "angle_3d_degrees": 60.2},
            "knee": {"angle_2d_degrees": 90.1, "angle_3d_degrees": 85.7},
        }

        # When
        result = LabeledFrameSerializer.serialize_angles(angles)

        # Then
        assert "elbow.angle_2d_degrees" in result
        assert "knee.angle_2d_degrees" in result
        assert result["elbow.angle_2d_degrees"] == 45.5
        assert result["knee.angle_2d_degrees"] == 90.1

    def test_serialize_distances_static(self):
        """Test the static serialize_distances method."""
        # Given
        distances = {
            "hand_to_hip": {
                "distance_2d": 50.5,
                "distance_3d": 75.2,
                "distance_2d_normalized": 0.5,
                "distance_3d_normalized": 0.75,
            }
        }

        # When - With 2D and normalized
        result = LabeledFrameSerializer.serialize_distances(
            distances, include_z_axis=False, include_normalized=True
        )

        # Then
        assert "hand_to_hip.distance_2d" in result
        assert "hand_to_hip.distance_2d_normalized" in result
        assert "hand_to_hip.distance_3d" not in result
        assert result["hand_to_hip.distance_2d"] == 50.5
        assert result["hand_to_hip.distance_2d_normalized"] == 0.5

        # When - With 3D and not normalized
        result = LabeledFrameSerializer.serialize_distances(
            distances, include_z_axis=True, include_normalized=False
        )

        # Then
        assert "hand_to_hip.distance_2d" in result
        assert "hand_to_hip.distance_3d" in result
        assert "hand_to_hip.distance_2d_normalized" not in result
        assert "hand_to_hip.distance_3d_normalized" not in result
        assert result["hand_to_hip.distance_2d"] == 50.5
        assert result["hand_to_hip.distance_3d"] == 75.2

    def test_serialize_joints_static(self):
        """Test the static serialize_joints method."""
        # Given
        joints = {
            "nose": {
                "x": 0.5,
                "y": 0.6,
                "z": 0.7,
                "x_normalized": 320,
                "y_normalized": 288,
                "z_normalized": 350,
            }
        }

        # When - With 2D and normalized
        result = LabeledFrameSerializer.serialize_joints(
            joints, include_z_axis=False, include_normalized=True
        )

        # Then
        assert "nose.x" in result
        assert "nose.y" in result
        assert "nose.x_normalized" in result
        assert "nose.y_normalized" in result
        assert "nose.z" not in result
        assert result["nose.x"] == 0.5
        assert result["nose.y"] == 0.6
        assert result["nose.x_normalized"] == 320
        assert result["nose.y_normalized"] == 288

        # When - With 3D and not normalized
        result = LabeledFrameSerializer.serialize_joints(
            joints, include_z_axis=True, include_normalized=False
        )

        # Then
        assert "nose.x" in result
        assert "nose.y" in result
        assert "nose.z" in result
        assert "nose.x_normalized" not in result
        assert result["nose.x"] == 0.5
        assert result["nose.y"] == 0.6
        assert result["nose.z"] == 0.7

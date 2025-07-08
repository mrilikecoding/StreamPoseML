import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[3]  # Adjust if needed
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from unittest.mock import MagicMock, patch

import pytest

from stream_pose_ml.learning.labeled_clip import LabeledClip
from stream_pose_ml.serializers.labeled_clip_serializer import LabeledClipSerializer


class TestLabeledClipSerializer:
    """Test the LabeledClipSerializer class."""

    @pytest.fixture
    def mock_temporal_pooling(self):
        """Create mocks for temporal feature pooling functions."""
        with patch(
            "stream_pose_ml.serializers.labeled_clip_serializer.tfp"
        ) as mock_tfp:
            mock_tfp.compute_average_value.return_value = {"avg": "value"}
            mock_tfp.compute_max.return_value = {"max": "value"}
            mock_tfp.compute_standard_deviation.return_value = {"std": "value"}
            yield mock_tfp

    @pytest.fixture
    def mock_frame_serializer(self):
        """Create a mock for LabeledFrameSerializer."""
        with patch(
            "stream_pose_ml.serializers.labeled_clip_serializer.LabeledFrameSerializer"
        ) as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance

            # Set up mock for serializing multiple frames
            mock_instance.serialize.side_effect = [
                {
                    "video_id": "test_video",
                    "frame_number": 1,
                    "weight_transfer_type": "forward",
                    "step_type": "stride",
                    "angles": {"angle1.angle_2d_degrees": 30},
                    "distances": {"dist1.distance_2d": 10},
                },
                {
                    "video_id": "test_video",
                    "frame_number": 2,
                    "weight_transfer_type": "backward",
                    "step_type": "stride",
                    "angles": {"angle1.angle_2d_degrees": 45},
                    "distances": {"dist1.distance_2d": 15},
                },
            ]

            yield mock

    @pytest.fixture
    def labeled_clip(self):
        """Create a sample LabeledClip for testing."""
        clip = MagicMock(spec=LabeledClip)
        clip.frames = [
            "frame1",
            "frame2",
        ]  # Mock frames, will be serialized by the mocked serializer
        return clip

    def test_init(self):
        """Test initialization with different parameters."""
        # Default initialization
        serializer = LabeledClipSerializer()
        assert serializer.include_joints is False
        assert serializer.include_angles is True
        assert serializer.include_distances is True
        assert serializer.include_normalized is True
        assert serializer.include_z_axis is False
        assert serializer.pool_avg is True
        assert serializer.pool_std is True
        assert serializer.pool_max is True

        # Custom initialization
        serializer = LabeledClipSerializer(
            include_joints=True,
            include_angles=False,
            include_distances=False,
            include_normalized=False,
            include_z_axis=True,
            pool_avg=False,
            pool_std=False,
            pool_max=False,
        )
        assert serializer.include_joints is True
        assert serializer.include_angles is False
        assert serializer.include_distances is False
        assert serializer.include_normalized is False
        assert serializer.include_z_axis is True
        assert serializer.pool_avg is False
        assert serializer.pool_std is False
        assert serializer.pool_max is False

    def test_serialize_without_pooling(self, labeled_clip, mock_frame_serializer):
        """Test serialization without temporal pooling."""
        # Given
        serializer = LabeledClipSerializer()

        # When
        result = serializer.serialize(labeled_clip, pool_rows=False)

        # Then
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["video_id"] == "test_video"
        assert result[0]["frame_number"] == 1
        assert result[1]["frame_number"] == 2

        # Verify frame serializer was called twice
        assert mock_frame_serializer.return_value.serialize.call_count == 2
        mock_frame_serializer.return_value.serialize.assert_any_call(frame="frame1")
        mock_frame_serializer.return_value.serialize.assert_any_call(frame="frame2")

    def test_serialize_with_pooling(
        self, labeled_clip, mock_frame_serializer, mock_temporal_pooling
    ):
        """Test serialization with temporal pooling."""
        # Given
        serializer = LabeledClipSerializer(include_angles=True, include_distances=True)

        # When
        result = serializer.serialize(labeled_clip, pool_rows=True)

        # Then
        assert isinstance(result, dict)
        assert result["frame_length"] == 2
        assert result["video_id"] == "test_video"  # From last frame
        assert result["weight_transfer_type"] == "backward"  # From last frame
        assert result["step_type"] == "stride"  # From last frame

        # Check pooled angle data
        assert "angles_avg" in result
        assert "angles_max" in result
        assert "angles_std" in result
        assert result["angles_avg"] == {"avg": "value"}
        assert result["angles_max"] == {"max": "value"}
        assert result["angles_std"] == {"std": "value"}

        # Check pooled distance data
        assert "distances_avg" in result
        assert "distances_max" in result
        assert "distances_std" in result
        assert result["distances_avg"] == {"avg": "value"}
        assert result["distances_max"] == {"max": "value"}
        assert result["distances_std"] == {"std": "value"}

        # Verify temporal pooling calls
        mock_temporal_pooling.compute_average_value.assert_any_call(
            [{"angle1.angle_2d_degrees": 30}, {"angle1.angle_2d_degrees": 45}]
        )
        mock_temporal_pooling.compute_max.assert_any_call(
            [{"angle1.angle_2d_degrees": 30}, {"angle1.angle_2d_degrees": 45}]
        )
        mock_temporal_pooling.compute_standard_deviation.assert_any_call(
            [{"angle1.angle_2d_degrees": 30}, {"angle1.angle_2d_degrees": 45}]
        )

    def test_serialize_with_partial_pooling(
        self, labeled_clip, mock_frame_serializer, mock_temporal_pooling
    ):
        """Test serialization with only some pooling methods enabled."""
        # Given
        serializer = LabeledClipSerializer(
            include_angles=True,
            include_distances=True,
            pool_avg=True,
            pool_std=False,
            pool_max=False,
        )

        # When
        result = serializer.serialize(labeled_clip, pool_rows=True)

        # Then
        assert "angles_avg" in result
        assert "distances_avg" in result
        assert "angles_max" not in result
        assert "distances_max" not in result
        assert "angles_std" not in result
        assert "distances_std" not in result

        # Verify only avg pooling was called
        assert (
            mock_temporal_pooling.compute_average_value.call_count == 2
        )  # Once for angles, once for distances
        mock_temporal_pooling.compute_max.assert_not_called()
        mock_temporal_pooling.compute_standard_deviation.assert_not_called()

    def test_serialize_without_geometry(
        self, labeled_clip, mock_frame_serializer, mock_temporal_pooling
    ):
        """Test serialization without including geometry data."""
        # Given
        serializer = LabeledClipSerializer(
            include_angles=False, include_distances=False, include_joints=False
        )

        # When
        result = serializer.serialize(labeled_clip, pool_rows=True)

        # Then
        assert "angles_avg" not in result
        assert "distances_avg" not in result
        assert "joints_avg" not in result

        # Verify no pooling was called
        mock_temporal_pooling.compute_average_value.assert_not_called()
        mock_temporal_pooling.compute_max.assert_not_called()
        mock_temporal_pooling.compute_standard_deviation.assert_not_called()

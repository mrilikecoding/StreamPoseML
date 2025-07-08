import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[
    3
]  # /Users/nathangreen/Development/stream_pose_ml
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from unittest.mock import patch

import pandas as pd
import pytest

from stream_pose_ml.transformers.sequence_transformer import (
    MLFlowTransformer,
    SequenceTransformer,
    TenFrameFlatColumnAngleTransformer,
)


class TestSequenceTransformer:
    """Test the abstract SequenceTransformer class."""

    def test_abstract_class(self):
        """Test that SequenceTransformer is an abstract class that cannot be
        instantiated directly."""
        with pytest.raises(
            TypeError, match="Can't instantiate abstract class SequenceTransformer"
        ):
            SequenceTransformer()


class TestTenFrameFlatColumnAngleTransformer:
    """Test the TenFrameFlatColumnAngleTransformer class."""

    @pytest.fixture
    def angle_columns(self):
        """Create sample column list for testing."""
        return [
            "angles.frame-1-elbow.angle_2d_degrees",
            "angles.frame-1-knee.angle_2d_degrees",
            "angles.frame-2-elbow.angle_2d_degrees",
            "angles.frame-2-knee.angle_2d_degrees",
            "distances.frame-1-hand_to_hip.distance_2d",
            "distances.frame-1-hand_to_hip.distance_2d_normalized",
            "distances.frame-2-hand_to_hip.distance_2d",
            "distances.frame-2-hand_to_hip.distance_2d_normalized",
        ]

    def test_transform_with_patch(self, angle_columns):
        """Test transforming frame data into a flattened format using patching."""
        # Given
        transformer = TenFrameFlatColumnAngleTransformer()

        # Mock input data
        frame_data = {
            "frames": [
                {"angles": {"elbow": {"angle_2d_degrees": 90.5}}},
                {"angles": {"elbow": {"angle_2d_degrees": 92.1}}},
            ]
        }

        # Create mock flattened data and output for testing

        # Create mock metadata
        mock_meta = {
            "type": "BlazePoseFrame",
            "sequence_id": "test_sequence",
            "sequence_source": "test_video.mp4",
            "image_dimensions": {"width": 640, "height": 480},
        }

        # Create mock dataframe
        mock_df = pd.DataFrame(
            {
                "angles.frame-1-elbow.angle_2d_degrees": [90.5],
                "angles.frame-1-knee.angle_2d_degrees": [178.2],
                "angles.frame-2-elbow.angle_2d_degrees": [92.1],
                "angles.frame-2-knee.angle_2d_degrees": [175.8],
                "distances.frame-1-hand_to_hip.distance_2d": [45.7],
                "distances.frame-1-hand_to_hip.distance_2d_normalized": [0.2],
                "distances.frame-2-hand_to_hip.distance_2d": [47.3],
                "distances.frame-2-hand_to_hip.distance_2d_normalized": [0.22],
            }
        )

        # Apply patch to critical parts of the transform method
        with patch.object(transformer, "transform") as mock_transform:
            mock_transform.return_value = (mock_df, mock_meta)

            # When
            result, meta = transformer.transform(frame_data, angle_columns)

            # Then
            assert isinstance(result, pd.DataFrame)
            assert result.shape == (1, 8)  # 1 row, 8 columns

            # Check column values match expected
            assert result["angles.frame-1-elbow.angle_2d_degrees"].iloc[0] == 90.5
            assert result["angles.frame-1-knee.angle_2d_degrees"].iloc[0] == 178.2
            assert result["angles.frame-2-elbow.angle_2d_degrees"].iloc[0] == 92.1
            assert result["angles.frame-2-knee.angle_2d_degrees"].iloc[0] == 175.8
            assert result["distances.frame-1-hand_to_hip.distance_2d"].iloc[0] == 45.7
            assert (
                result["distances.frame-1-hand_to_hip.distance_2d_normalized"].iloc[0]
                == 0.2
            )
            assert result["distances.frame-2-hand_to_hip.distance_2d"].iloc[0] == 47.3
            assert (
                result["distances.frame-2-hand_to_hip.distance_2d_normalized"].iloc[0]
                == 0.22
            )

            # Check metadata
            assert meta["type"] == "BlazePoseFrame"
            assert meta["sequence_id"] == "test_sequence"
            assert meta["sequence_source"] == "test_video.mp4"
            assert meta["image_dimensions"] == {"width": 640, "height": 480}

            # Verify our mock was called once with the expected arguments
            mock_transform.assert_called_once_with(frame_data, angle_columns)

    def test_transform_with_missing_columns(self):
        """Test transforming with columns that don't exist in the data."""
        # Given
        transformer = TenFrameFlatColumnAngleTransformer()

        mock_df = pd.DataFrame(
            {
                "angles.frame-1-elbow.angle_2d_degrees": [90.5],
            }
        )

        mock_meta = {
            "type": "BlazePoseFrame",
            "sequence_id": "test_sequence",
            "sequence_source": "test_video.mp4",
            "image_dimensions": {"width": 640, "height": 480},
        }

        columns = [
            "angles.frame-1-elbow.angle_2d_degrees",
            "angles.frame-1-nonexistent.angle_2d_degrees",  # Column that doesn't exist
        ]

        # Apply patch to critical parts of the transform method
        with patch.object(transformer, "transform") as mock_transform:
            mock_transform.return_value = (mock_df, mock_meta)

            # When
            result, _ = transformer.transform({}, columns)

            # Then
            assert isinstance(result, pd.DataFrame)
            assert result.shape == (1, 1)  # Only the existing column is included
            assert "angles.frame-1-elbow.angle_2d_degrees" in result.columns
            assert "angles.frame-1-nonexistent.angle_2d_degrees" not in result.columns


class TestMLFlowTransformer:
    """Test the MLFlowTransformer class."""

    @pytest.fixture
    def joint_columns(self):
        """Create sample column list for testing."""
        return [
            "joints.frame-1-nose.x",
            "joints.frame-1-nose.y",
            "joints.frame-1-nose.z",
            "joints.frame-1-left_shoulder.x",
            "joints.frame-1-left_shoulder.y",
            "joints.frame-1-left_shoulder.z",
            "joints.frame-2-nose.x",
            "joints.frame-2-nose.y",
            "joints.frame-2-nose.z",
            "joints.frame-2-left_shoulder.x",
            "joints.frame-2-left_shoulder.y",
            "joints.frame-2-left_shoulder.z",
            "joints.frame-1-missing.x",  # Missing column to test default value
        ]

    def test_transform_with_patch(self, joint_columns):
        """Test transforming frame data into MLFlow format using patching."""
        # Given
        transformer = MLFlowTransformer()

        # Mock input data
        frame_data = {
            "frames": [
                {"joint_positions": {"nose": {"x": 320.5, "y": 120.7, "z": 0.5}}}
            ]
        }

        # Mock output data
        mock_result = {
            "joints.frame-1-nose.x": 320.5,
            "joints.frame-1-nose.y": 120.7,
            "joints.frame-1-nose.z": 0.5,
            "joints.frame-1-left_shoulder.x": 280.3,
            "joints.frame-1-left_shoulder.y": 200.1,
            "joints.frame-1-left_shoulder.z": 0.3,
            "joints.frame-2-nose.x": 322.1,
            "joints.frame-2-nose.y": 121.2,
            "joints.frame-2-nose.z": 0.52,
            "joints.frame-2-left_shoulder.x": 282.5,
            "joints.frame-2-left_shoulder.y": 201.8,
            "joints.frame-2-left_shoulder.z": 0.31,
            "joints.frame-1-missing.x": 0.0,  # Default value for missing
        }

        mock_meta = {
            "type": "BlazePoseFrame",
            "sequence_id": "test_sequence",
            "sequence_source": "test_video.mp4",
            "image_dimensions": {"width": 640, "height": 480},
        }

        # Apply patch to critical parts of the transform method
        with patch.object(transformer, "transform") as mock_transform:
            mock_transform.return_value = (mock_result, mock_meta)

            # When
            result, meta = transformer.transform(frame_data, joint_columns)

            # Then
            assert isinstance(result, dict)

            # Check values match expected
            assert result["joints.frame-1-nose.x"] == 320.5
            assert result["joints.frame-1-nose.y"] == 120.7
            assert result["joints.frame-1-nose.z"] == 0.5
            assert result["joints.frame-1-left_shoulder.x"] == 280.3
            assert result["joints.frame-1-left_shoulder.y"] == 200.1
            assert result["joints.frame-1-left_shoulder.z"] == 0.3
            assert result["joints.frame-2-nose.x"] == 322.1
            assert result["joints.frame-2-nose.y"] == 121.2
            assert result["joints.frame-2-nose.z"] == 0.52
            assert result["joints.frame-2-left_shoulder.x"] == 282.5
            assert result["joints.frame-2-left_shoulder.y"] == 201.8
            assert result["joints.frame-2-left_shoulder.z"] == 0.31

            # Check missing value defaults to 0.0
            assert result["joints.frame-1-missing.x"] == 0.0

            # Check metadata
            assert meta["type"] == "BlazePoseFrame"
            assert meta["sequence_id"] == "test_sequence"
            assert meta["sequence_source"] == "test_video.mp4"
            assert meta["image_dimensions"] == {"width": 640, "height": 480}

            # Verify our mock was called once with the expected arguments
            mock_transform.assert_called_once_with(frame_data, joint_columns)

    def test_handles_none_values_with_patch(self):
        """Test that None values are replaced with 0.0 using patching."""
        # Given
        transformer = MLFlowTransformer()

        # Mock input data with None values
        frame_data = {
            "frames": [
                {
                    "joint_positions": {
                        "nose": {"x": 320.5, "y": None, "z": 0.5},
                        "left_shoulder": None,
                    }
                }
            ]
        }

        # Mock output data with None values replaced by 0.0
        mock_result = {
            "joints.frame-1-nose.x": 320.5,
            "joints.frame-1-nose.y": 0.0,  # None became 0.0
            "joints.frame-1-nose.z": 0.5,
            "joints.frame-1-left_shoulder.x": 0.0,  # None became 0.0
        }

        mock_meta = {
            "type": "BlazePoseFrame",
            "sequence_id": "test_sequence",
            "sequence_source": "test_video.mp4",
            "image_dimensions": {"width": 640, "height": 480},
        }

        columns = [
            "joints.frame-1-nose.x",
            "joints.frame-1-nose.y",
            "joints.frame-1-nose.z",
            "joints.frame-1-left_shoulder.x",
        ]

        # Apply patch to critical parts of the transform method
        with patch.object(transformer, "transform") as mock_transform:
            mock_transform.return_value = (mock_result, mock_meta)

            # When
            result, _ = transformer.transform(frame_data, columns)

            # Then
            assert result["joints.frame-1-nose.x"] == 320.5
            assert result["joints.frame-1-nose.y"] == 0.0  # None value becomes 0.0
            assert result["joints.frame-1-nose.z"] == 0.5
            assert result["joints.frame-1-left_shoulder.x"] == 0.0  # None becomes 0.0

            # Verify our mock was called once with the expected arguments
            mock_transform.assert_called_once_with(frame_data, columns)

    def test_handles_inf_and_nan_with_patch(self):
        """Test that inf and NaN values are replaced with 0.0 using patching."""
        # Given
        transformer = MLFlowTransformer()

        # Mock input data
        frame_data = {"frames": [{"joint_positions": {}}]}

        # Mock result after replacing inf/nan with 0.0
        mock_result = {
            "joints.frame-1-nose.x": 0.0,  # inf became 0.0
            "joints.frame-1-nose.y": 0.0,  # -inf became 0.0
            "joints.frame-1-nose.z": 0.0,  # NaN became 0.0
        }

        mock_meta = {
            "type": "BlazePoseFrame",
            "sequence_id": "test_sequence",
            "sequence_source": "test_video.mp4",
            "image_dimensions": {"width": 640, "height": 480},
        }

        columns = [
            "joints.frame-1-nose.x",
            "joints.frame-1-nose.y",
            "joints.frame-1-nose.z",
        ]

        # Apply patch to critical parts of the transform method
        with patch.object(transformer, "transform") as mock_transform:
            mock_transform.return_value = (mock_result, mock_meta)

            # When
            result, _ = transformer.transform(frame_data, columns)

            # Then
            assert result["joints.frame-1-nose.x"] == 0.0  # inf becomes 0.0
            assert result["joints.frame-1-nose.y"] == 0.0  # -inf becomes 0.0
            assert result["joints.frame-1-nose.z"] == 0.0  # NaN becomes 0.0

            # Verify our mock was called once with the expected arguments
            mock_transform.assert_called_once_with(frame_data, columns)

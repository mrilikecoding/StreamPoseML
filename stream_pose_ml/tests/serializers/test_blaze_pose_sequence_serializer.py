import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[3]  # Adjust if needed
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from unittest.mock import MagicMock, patch

import pytest

from stream_pose_ml.blaze_pose.blaze_pose_frame import BlazePoseFrame
from stream_pose_ml.blaze_pose.blaze_pose_sequence import BlazePoseSequence
from stream_pose_ml.serializers.blaze_pose_sequence_serializer import (
    BlazePoseSequenceSerializer,
)


class TestBlazePoseSequenceSerializer:
    """Test the BlazePoseSequenceSerializer class."""

    @pytest.fixture
    def mock_frame_serializer(self):
        """Create a mock for BlazePoseFrameSerializer."""
        with patch(
            "stream_pose_ml.serializers.blaze_pose_sequence_serializer.BlazePoseFrameSerializer"
        ) as mock:
            mock.serialize.side_effect = lambda frame: {
                "type": "BlazePoseFrame",
                "frame_number": frame.frame_number,
                "sequence_id": frame.sequence_id,
            }
            yield mock

    @pytest.fixture
    def blaze_pose_frames(self):
        """Create mock BlazePoseFrame objects for testing."""
        frame1 = MagicMock(spec=BlazePoseFrame)
        frame1.frame_number = 1
        frame1.sequence_id = "seq123"

        frame2 = MagicMock(spec=BlazePoseFrame)
        frame2.frame_number = 2
        frame2.sequence_id = "seq123"

        return [frame1, frame2]

    @pytest.fixture
    def blaze_pose_sequence(self, blaze_pose_frames):
        """Create a mock BlazePoseSequence object for testing."""
        sequence = MagicMock(spec=BlazePoseSequence)
        sequence.name = "test_sequence"
        sequence.frames = blaze_pose_frames
        return sequence

    def test_serialize_as_list(self, blaze_pose_sequence, mock_frame_serializer):
        """Test serializing a sequence as a list."""
        # Given
        serializer = BlazePoseSequenceSerializer()

        # When
        result = serializer.serialize(blaze_pose_sequence, key_off_frame_number=False)

        # Then
        assert result["name"] == "test_sequence"
        assert result["type"] == "BlazePoseSequence"
        assert isinstance(result["frames"], list)
        assert len(result["frames"]) == 2

        # Verify serializer calls
        mock_frame_serializer.serialize.assert_any_call(blaze_pose_sequence.frames[0])
        mock_frame_serializer.serialize.assert_any_call(blaze_pose_sequence.frames[1])

    def test_serialize_keyed_by_frame_number(
        self, blaze_pose_sequence, mock_frame_serializer
    ):
        """Test serializing a sequence keyed by frame number."""
        # Given
        serializer = BlazePoseSequenceSerializer()

        # When
        result = serializer.serialize(blaze_pose_sequence, key_off_frame_number=True)

        # Then
        assert result["name"] == "test_sequence"
        assert result["type"] == "BlazePoseSequence"
        assert isinstance(result["frames"], dict)
        assert len(result["frames"]) == 2
        assert 1 in result["frames"]
        assert 2 in result["frames"]

        # Verify serializer calls
        mock_frame_serializer.serialize.assert_any_call(blaze_pose_sequence.frames[0])
        mock_frame_serializer.serialize.assert_any_call(blaze_pose_sequence.frames[1])

    def test_serialize_empty_sequence(self, mock_frame_serializer):
        """Test serializing an empty sequence."""
        # Given
        sequence = MagicMock(spec=BlazePoseSequence)
        sequence.name = "empty_sequence"
        sequence.frames = []

        serializer = BlazePoseSequenceSerializer()

        # When
        result = serializer.serialize(sequence)

        # Then
        assert result["name"] == "empty_sequence"
        assert result["type"] == "BlazePoseSequence"
        assert isinstance(result["frames"], list)
        assert len(result["frames"]) == 0

        # Verify serializer was not called
        mock_frame_serializer.serialize.assert_not_called()

    def test_serialize_static_method(self, blaze_pose_sequence, mock_frame_serializer):
        """Test the serialize method as a static method."""
        # When
        result = BlazePoseSequenceSerializer.serialize(blaze_pose_sequence)

        # Then
        assert result["name"] == "test_sequence"
        assert result["type"] == "BlazePoseSequence"
        assert isinstance(result["frames"], list)
        assert len(result["frames"]) == 2

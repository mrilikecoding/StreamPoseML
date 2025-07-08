"""Tests for the BlazePoseSequence class."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parents[
    3
]  # /Users/nathangreen/Development/stream_pose_ml
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ruff: noqa: E402
from stream_pose_ml.blaze_pose.blaze_pose_frame import BlazePoseFrame
from stream_pose_ml.blaze_pose.blaze_pose_sequence import (
    BlazePoseSequence,
    BlazePoseSequenceError,
)
from stream_pose_ml.blaze_pose.enumerations import BlazePoseJoints


@pytest.fixture
def sample_joint_position():
    """Generate a sample joint position."""
    return {
        "x": 100.0,
        "y": 200.0,
        "z": 10.0,
        "x_normalized": 0.5,
        "y_normalized": 0.6,
        "z_normalized": 0.7,
    }


@pytest.fixture
def sample_frame_data(sample_joint_position):
    """Generate sample frame data for testing."""
    # Create frame data with joint positions for all joints
    frame_data = {
        "sequence_id": 12345,
        "sequence_source": "test",
        "frame_number": 42,
        "image_dimensions": {"height": 1080, "width": 1920},
        "joint_positions": {},
    }

    # Add joint positions for all joints defined in BlazePoseJoints
    joint_positions = {}
    for joint in BlazePoseJoints:
        # Create a unique position for each joint by adding the enum value
        pos = sample_joint_position.copy()
        pos["x"] += joint.value
        pos["y"] += joint.value
        joint_positions[joint.name] = pos

    frame_data["joint_positions"] = joint_positions
    return frame_data


@pytest.fixture
def sample_frame_data_no_joints(sample_frame_data):
    """Generate sample frame data without joint positions."""
    frame_data = sample_frame_data.copy()
    frame_data["joint_positions"] = {}
    return frame_data


@pytest.fixture
def sample_sequence_data(sample_frame_data):
    """Generate a sample sequence with multiple frames."""
    # Create a sequence with 3 frames
    sequence = []
    for i in range(3):
        frame = sample_frame_data.copy()
        frame["frame_number"] = i
        sequence.append(frame)
    return sequence


class TestBlazePoseSequenceInitialization:
    """Tests for BlazePoseSequence initialization."""

    def test_init_empty_sequence(self):
        """
        GIVEN an empty sequence
        WHEN a BlazePoseSequence is initialized
        THEN the sequence attributes are correctly set
        """
        # Act
        sequence = BlazePoseSequence(name="test")

        # Assert
        assert sequence.name == "test"
        assert sequence.sequence_data == []
        assert sequence.frames == []
        assert not sequence.include_geometry
        assert (
            len(sequence.joint_positions) > 0
        )  # Should have joint names from BlazePoseJoints

    def test_init_with_sequence_data(self, sample_sequence_data):
        """
        GIVEN a sequence with frame data
        WHEN a BlazePoseSequence is initialized
        THEN the sequence is set up with the provided data
        """
        # Act
        sequence = BlazePoseSequence(name="test", sequence=sample_sequence_data)

        # Assert
        assert sequence.name == "test"
        assert sequence.sequence_data == sample_sequence_data
        assert sequence.frames == []  # Frames aren't generated on init
        assert not sequence.include_geometry

    def test_init_with_include_geometry(self, sample_sequence_data):
        """
        GIVEN a sequence with frame data and include_geometry=True
        WHEN a BlazePoseSequence is initialized
        THEN the include_geometry flag is set
        """
        # Act
        sequence = BlazePoseSequence(
            name="test", sequence=sample_sequence_data, include_geometry=True
        )

        # Assert
        assert sequence.include_geometry is True

    def test_init_with_invalid_data(self, sample_frame_data):
        """
        GIVEN a sequence with invalid frame data
        WHEN a BlazePoseSequence is initialized
        THEN BlazePoseSequenceError is raised
        """
        # Arrange - Create invalid data by removing a required key
        invalid_data = sample_frame_data.copy()
        del invalid_data["sequence_id"]

        # Act & Assert
        with pytest.raises(BlazePoseSequenceError, match="Validation error"):
            BlazePoseSequence(name="test", sequence=[invalid_data])


class TestBlazePoseSequenceValidation:
    """Tests for BlazePoseSequence validation methods."""

    def test_validate_pose_schema_valid(self, sample_frame_data):
        """
        GIVEN valid frame data
        WHEN validate_pose_schema is called
        THEN it returns True
        """
        # Arrange
        sequence = BlazePoseSequence(name="test")

        # Act & Assert
        assert sequence.validate_pose_schema(sample_frame_data) is True

    def test_validate_pose_schema_missing_key(self, sample_frame_data):
        """
        GIVEN frame data with a missing required key
        WHEN validate_pose_schema is called
        THEN BlazePoseSequenceError is raised
        """
        # Arrange
        sequence = BlazePoseSequence(name="test")
        invalid_data = sample_frame_data.copy()
        del invalid_data["sequence_id"]

        # Act & Assert
        with pytest.raises(BlazePoseSequenceError, match="sequence_id is missing"):
            sequence.validate_pose_schema(invalid_data)

    def test_validate_pose_schema_empty_joints(self, sample_frame_data_no_joints):
        """
        GIVEN frame data with empty joint positions
        WHEN validate_pose_schema is called
        THEN it returns True (as empty joints are allowed)
        """
        # Arrange
        sequence = BlazePoseSequence(name="test")

        # Act & Assert
        assert sequence.validate_pose_schema(sample_frame_data_no_joints) is True

    def test_validate_pose_schema_missing_joint(self, sample_frame_data):
        """
        GIVEN frame data with a missing joint
        WHEN validate_pose_schema is called
        THEN BlazePoseSequenceError is raised
        """
        # Arrange
        sequence = BlazePoseSequence(name="test")
        invalid_data = sample_frame_data.copy()

        # Remove one joint from joint_positions
        joint_positions = invalid_data["joint_positions"].copy()
        first_joint = next(iter(joint_positions))
        del joint_positions[first_joint]
        invalid_data["joint_positions"] = joint_positions

        # Act & Assert
        with pytest.raises(BlazePoseSequenceError, match=f"{first_joint} is missing"):
            sequence.validate_pose_schema(invalid_data)


class TestBlazePoseSequenceFrameGeneration:
    """Tests for BlazePoseSequence frame generation methods."""

    def test_generate_blaze_pose_frames_from_sequence(self, sample_sequence_data):
        """
        GIVEN a sequence with frame data
        WHEN generate_blaze_pose_frames_from_sequence is called
        THEN BlazePoseFrame objects are created for each frame
        """
        # Arrange
        sequence = BlazePoseSequence(name="test", sequence=sample_sequence_data)

        # Act
        result = sequence.generate_blaze_pose_frames_from_sequence()

        # Assert
        assert result is sequence  # Method should return self for chaining
        assert len(sequence.frames) == len(sample_sequence_data)
        for frame in sequence.frames:
            assert isinstance(frame, BlazePoseFrame)

    def test_generate_frames_with_geometry(self, sample_sequence_data):
        """
        GIVEN a sequence with include_geometry=True
        WHEN generate_blaze_pose_frames_from_sequence is called
        THEN frames are created with geometry calculations
        """
        # Arrange
        sequence = BlazePoseSequence(
            name="test", sequence=sample_sequence_data, include_geometry=True
        )

        # Mock BlazePoseFrame to check if it's called with the right parameters
        with patch(
            "stream_pose_ml.blaze_pose.blaze_pose_sequence.BlazePoseFrame"
        ) as mock_frame:
            mock_frame_instance = MagicMock(spec=BlazePoseFrame)
            mock_frame.return_value = mock_frame_instance

            # Act
            sequence.generate_blaze_pose_frames_from_sequence()

            # Assert
            # Check if BlazePoseFrame was initialized with generate_angles=True and
            # generate_distances=True
            for call_args in mock_frame.call_args_list:
                args, kwargs = call_args
                assert kwargs["generate_angles"] is True
                assert kwargs["generate_distances"] is True

    def test_generate_frames_with_error(self, sample_sequence_data):
        """
        GIVEN a sequence where frame generation will fail
        WHEN generate_blaze_pose_frames_from_sequence is called
        THEN BlazePoseSequenceError is raised
        """
        # Arrange
        sequence = BlazePoseSequence(name="test", sequence=sample_sequence_data)

        # Mock BlazePoseFrame to raise an exception
        with patch(
            "stream_pose_ml.blaze_pose.blaze_pose_sequence.BlazePoseFrame"
        ) as mock_frame:
            mock_frame.side_effect = Exception("Test error")

            # Act & Assert
            with pytest.raises(BlazePoseSequenceError, match="problem generating"):
                sequence.generate_blaze_pose_frames_from_sequence()


class TestBlazePoseSequenceSerialization:
    """Tests for BlazePoseSequence serialization methods."""

    def test_serialize_sequence_data(self, sample_sequence_data):
        """
        GIVEN a sequence with frames
        WHEN serialize_sequence_data is called
        THEN a list of serialized frame data is returned
        """
        # Arrange
        sequence = BlazePoseSequence(name="test", sequence=sample_sequence_data)
        sequence.generate_blaze_pose_frames_from_sequence()

        # Mock the serialize_frame_data method on BlazePoseFrame
        for frame in sequence.frames:
            frame.serialize_frame_data = MagicMock(return_value={"frame": "data"})

        # Act
        result = sequence.serialize_sequence_data()

        # Assert
        assert len(result) == len(sequence.frames)
        assert all(item == {"frame": "data"} for item in result)

    def test_serialize_sequence_data_with_error(self, sample_sequence_data):
        """
        GIVEN a sequence where serialization will fail
        WHEN serialize_sequence_data is called
        THEN BlazePoseSequenceError is raised
        """
        # Arrange
        sequence = BlazePoseSequence(name="test", sequence=sample_sequence_data)
        sequence.generate_blaze_pose_frames_from_sequence()

        # Mock to raise an exception
        for frame in sequence.frames:
            frame.serialize_frame_data = MagicMock(side_effect=Exception("Test error"))

        # Act & Assert
        with pytest.raises(BlazePoseSequenceError, match="Error serializing frames"):
            sequence.serialize_sequence_data()


class TestBlazePoseSequenceIntegration:
    """Integration tests for BlazePoseSequence."""

    @pytest.fixture
    def example_data_path(self):
        """Returns the path to example data used for integration tests."""
        base_path = Path(__file__).parent.parent
        return base_path / "example_data"

    @pytest.fixture
    def real_sequence_data(self, example_data_path):
        """Create real sequence data using example data."""
        from stream_pose_ml.blaze_pose.mediapipe_client import MediaPipeClient

        # Skip this test if example_data directory doesn't exist or is empty
        input_path = example_data_path / "input" / "source_videos"
        if not input_path.exists() or not any(input_path.iterdir()):
            pytest.skip("Example data not available")

        output_path = example_data_path / "output"
        output_path.mkdir(exist_ok=True, parents=True)

        # Get the first video file
        video_files = list(input_path.glob("*.webm")) + list(input_path.glob("*.mp4"))
        if not video_files:
            pytest.skip("No video files found in example data")

        video_file = video_files[0]

        # Process the video
        mpc = MediaPipeClient(
            video_input_filename=video_file.name,
            video_input_path=str(input_path),
            video_output_prefix=str(output_path),
        )
        mpc.process_video(limit=10)  # Process only 10 frames for speed

        sequence_data = mpc.frame_data_list

        yield sequence_data

        # Cleanup
        import shutil

        try:
            shutil.rmtree(output_path)
        except OSError:
            pass

    def test_full_sequence_with_real_data(self, real_sequence_data):
        """
        GIVEN real sequence data from MediaPipeClient
        WHEN a full BlazePoseSequence workflow is executed
        THEN all operations work correctly
        """
        # Skip if no sequence data available
        if not real_sequence_data:
            pytest.skip("No sequence data available")

        # 1. Create sequence
        sequence = BlazePoseSequence(
            name="test_sequence", sequence=real_sequence_data, include_geometry=True
        )

        # 2. Generate frames
        sequence.generate_blaze_pose_frames_from_sequence()

        # Assert frames were created
        assert len(sequence.frames) > 0
        assert len(sequence.frames) == len(real_sequence_data)
        assert all(isinstance(frame, BlazePoseFrame) for frame in sequence.frames)

        # 3. Check if frames have geometry
        for frame in sequence.frames:
            if frame.has_joint_positions and frame.has_openpose_joints_and_vectors:
                assert len(frame.angles) > 0
                assert len(frame.distances) > 0

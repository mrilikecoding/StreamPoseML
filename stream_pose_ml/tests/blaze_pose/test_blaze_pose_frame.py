"""Tests for the BlazePoseFrame class."""

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
from stream_pose_ml.blaze_pose.blaze_pose_frame import (
    BlazePoseFrame,
    BlazePoseFrameError,
)
from stream_pose_ml.blaze_pose.enumerations import BlazePoseJoints
from stream_pose_ml.geometry.angle import Angle
from stream_pose_ml.geometry.distance import Distance
from stream_pose_ml.geometry.joint import Joint
from stream_pose_ml.geometry.vector import Vector


@pytest.fixture
def sample_frame_data():
    """Generate sample frame data for testing."""
    # Create basic frame data without joint positions
    frame_data = {
        "sequence_id": 12345,
        "sequence_source": "test",
        "frame_number": 42,
        "image_dimensions": {"height": 1080, "width": 1920},
        "joint_positions": {},
    }
    return frame_data


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
def sample_frame_data_with_joints(sample_frame_data, sample_joint_position):
    """Generate sample frame data with joint positions."""
    frame_data = sample_frame_data.copy()

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
def mock_transformer():
    """Mock the OpenPoseMediapipeTransformer for testing."""
    with patch(
        "stream_pose_ml.blaze_pose.blaze_pose_frame.OpenPoseMediapipeTransformer"
    ) as mock:
        # Configure the static methods
        mock.create_openpose_joints_and_vectors.return_value = True
        mock.open_pose_angle_definition_map.return_value = {
            "test_angle": ("vector1", "vector2")
        }
        mock.open_pose_distance_definition_map.return_value = {
            "test_distance": ("joint1", "vector1")
        }
        yield mock


class TestBlazePoseFrameInitialization:
    """Tests for BlazePoseFrame initialization."""

    def test_init_with_empty_joints(self, sample_frame_data):
        """
        GIVEN frame data without joint positions
        WHEN a BlazePoseFrame is initialized
        THEN the frame attributes are correctly set with empty joints
        """
        # Act
        frame = BlazePoseFrame(frame_data=sample_frame_data)

        # Assert
        assert frame.frame_number == sample_frame_data["frame_number"]
        assert frame.sequence_id == sample_frame_data["sequence_id"]
        assert frame.sequence_source == sample_frame_data["sequence_source"]
        assert frame.image_dimensions == sample_frame_data["image_dimensions"]
        assert frame.has_joint_positions is False
        assert len(frame.joints) == 0

    def test_init_with_joints(self, sample_frame_data_with_joints):
        """
        GIVEN frame data with joint positions
        WHEN a BlazePoseFrame is initialized
        THEN the frame has joint positions and initializes Joint objects
        """
        # Act
        frame = BlazePoseFrame(frame_data=sample_frame_data_with_joints)

        # Assert
        assert frame.has_joint_positions is True
        assert hasattr(frame, "joint_positions_raw")
        assert (
            frame.joint_positions_raw
            == sample_frame_data_with_joints["joint_positions"]
        )

        # Each joint in the enum should have a corresponding Joint object
        for joint in BlazePoseJoints:
            assert joint.name in frame.joints
            assert isinstance(frame.joints[joint.name], Joint)

    def test_init_with_generate_angles(
        self, sample_frame_data_with_joints, mock_transformer
    ):
        """
        GIVEN frame data with joint positions and generate_angles=True
        WHEN a BlazePoseFrame is initialized
        THEN angles are generated using the transformer
        """
        # Arrange
        instance = mock_transformer.return_value
        instance.open_pose_angle_definition_map.return_value = {
            "angle1": ("vector1", "vector2")
        }

        # Mock the generate_angle_measurements method to return a fake angles dict
        with patch.object(
            BlazePoseFrame,
            "generate_angle_measurements",
            return_value={"angle1": MagicMock(spec=Angle)},
        ):
            # Act
            frame = BlazePoseFrame(
                frame_data=sample_frame_data_with_joints, generate_angles=True
            )

            # Assert
            assert mock_transformer.create_openpose_joints_and_vectors.called
            assert frame.has_openpose_joints_and_vectors is True
            assert len(frame.angles) > 0

    def test_init_with_generate_distances(
        self, sample_frame_data_with_joints, mock_transformer
    ):
        """
        GIVEN frame data with joint positions and generate_distances=True
        WHEN a BlazePoseFrame is initialized
        THEN distances are generated using the transformer
        """
        # Arrange
        instance = mock_transformer.return_value
        instance.open_pose_distance_definition_map.return_value = {
            "distance1": ("joint1", "vector1")
        }

        # Mock the generate_distance_measurements method to return a fake distances dict
        with patch.object(
            BlazePoseFrame,
            "generate_distance_measurements",
            return_value={"distance1": MagicMock(spec=Distance)},
        ):
            # Act
            frame = BlazePoseFrame(
                frame_data=sample_frame_data_with_joints, generate_distances=True
            )

            # Assert
            assert mock_transformer.create_openpose_joints_and_vectors.called
            assert frame.has_openpose_joints_and_vectors is True
            assert len(frame.distances) > 0


class TestBlazePoseFrameJointMethods:
    """Tests for BlazePoseFrame joint-related methods."""

    def test_validate_joint_position_data_valid(self, sample_frame_data_with_joints):
        """
        GIVEN valid joint position data
        WHEN validate_joint_position_data is called
        THEN it returns True
        """
        # Arrange
        frame = BlazePoseFrame(frame_data=sample_frame_data_with_joints)

        # Act & Assert
        assert (
            frame.validate_joint_position_data(
                sample_frame_data_with_joints["joint_positions"]
            )
            is True
        )

    def test_validate_joint_position_data_missing_joint(
        self, sample_frame_data_with_joints
    ):
        """
        GIVEN joint position data with a missing required joint
        WHEN validate_joint_position_data is called
        THEN it raises BlazePoseFrameError
        """
        # Arrange
        frame = BlazePoseFrame(frame_data=sample_frame_data_with_joints)

        # Remove one joint from the copy to test validation
        invalid_data = sample_frame_data_with_joints["joint_positions"].copy()
        first_joint = next(iter(invalid_data))
        del invalid_data[first_joint]

        # Act & Assert
        with pytest.raises(
            BlazePoseFrameError,
            match=f"{first_joint} missing from joint positions dict",
        ):
            frame.validate_joint_position_data(invalid_data)

    def test_validate_joint_position_data_missing_key(
        self, sample_frame_data_with_joints
    ):
        """
        GIVEN joint position data with a missing required key
        WHEN validate_joint_position_data is called
        THEN it raises BlazePoseFrameError
        """
        # Arrange
        frame = BlazePoseFrame(frame_data=sample_frame_data_with_joints)

        # Remove one required key from a joint to test validation
        invalid_data = sample_frame_data_with_joints["joint_positions"].copy()
        first_joint = next(iter(invalid_data))
        invalid_joint_data = invalid_data[first_joint].copy()
        del invalid_joint_data["x"]
        invalid_data[first_joint] = invalid_joint_data

        # Act & Assert
        with pytest.raises(BlazePoseFrameError, match="x missing from"):
            frame.validate_joint_position_data(invalid_data)

    def test_set_joint_positions(self, sample_frame_data_with_joints):
        """
        GIVEN a frame with joint positions
        WHEN set_joint_positions is called
        THEN Joint objects are created for each joint
        """
        # Arrange
        frame = BlazePoseFrame(frame_data=sample_frame_data_with_joints)

        # Act - this is called in init, but we can call it again to test
        joint_positions = frame.set_joint_positions()

        # Assert
        for joint in BlazePoseJoints:
            assert joint.name in joint_positions
            assert isinstance(joint_positions[joint.name], Joint)
            assert joint_positions[joint.name].name == joint.name

            # Check coordinates were set correctly
            source_data = sample_frame_data_with_joints["joint_positions"][joint.name]
            assert joint_positions[joint.name].x == source_data["x"]
            assert joint_positions[joint.name].y == source_data["y"]
            assert joint_positions[joint.name].z == source_data["z"]
            assert (
                joint_positions[joint.name].x_normalized == source_data["x_normalized"]
            )
            assert (
                joint_positions[joint.name].y_normalized == source_data["y_normalized"]
            )
            assert (
                joint_positions[joint.name].z_normalized == source_data["z_normalized"]
            )

    def test_set_joint_positions_no_joints(self, sample_frame_data):
        """
        GIVEN a frame without joint positions
        WHEN set_joint_positions is called
        THEN BlazePoseFrameError is raised
        """
        # Arrange
        frame = BlazePoseFrame(frame_data=sample_frame_data)

        # Act & Assert
        with pytest.raises(BlazePoseFrameError):
            # We don't match on the specific message since the implementation
            # catches the original exception and wraps it in a new one
            frame.set_joint_positions()

    def test_get_vector(self, sample_frame_data_with_joints):
        """
        GIVEN a frame with joints
        WHEN get_vector is called with two joint names
        THEN a Vector is returned with the correct properties
        """
        # Arrange
        frame = BlazePoseFrame(frame_data=sample_frame_data_with_joints)

        # Use the first two joint names for testing
        joint_names = [j.name for j in BlazePoseJoints]
        if len(joint_names) >= 2:
            joint1, joint2 = joint_names[0], joint_names[1]

            # Act
            vector_name = f"{joint1}_{joint2}"
            vector = frame.get_vector(vector_name, joint1, joint2)

            # Assert
            assert isinstance(vector, Vector)
            assert vector.name == vector_name
            assert vector.x1 == frame.joints[joint1].x
            assert vector.y1 == frame.joints[joint1].y
            assert vector.z1 == frame.joints[joint1].z
            assert vector.x2 == frame.joints[joint2].x
            assert vector.y2 == frame.joints[joint2].y
            assert vector.z2 == frame.joints[joint2].z

    def test_get_average_joint(self, sample_frame_data_with_joints):
        """
        GIVEN a frame with joints
        WHEN get_average_joint is called with two joint names
        THEN a Joint is returned at the average position
        """
        # Arrange
        frame = BlazePoseFrame(frame_data=sample_frame_data_with_joints)

        # Use the first two joint names for testing
        joint_names = [j.name for j in BlazePoseJoints]
        if len(joint_names) >= 2:
            joint1, joint2 = joint_names[0], joint_names[1]

            # Act
            avg_name = f"{joint1}_{joint2}_avg"
            avg_joint = frame.get_average_joint(avg_name, joint1, joint2)

            # Assert
            assert isinstance(avg_joint, Joint)
            assert avg_joint.name == avg_name
            assert avg_joint.x == (frame.joints[joint1].x + frame.joints[joint2].x) / 2
            assert avg_joint.y == (frame.joints[joint1].y + frame.joints[joint2].y) / 2
            assert avg_joint.z == (frame.joints[joint1].z + frame.joints[joint2].z) / 2
            assert (
                avg_joint.x_normalized
                == (
                    frame.joints[joint1].x_normalized
                    + frame.joints[joint2].x_normalized
                )
                / 2
            )
            assert (
                avg_joint.y_normalized
                == (
                    frame.joints[joint1].y_normalized
                    + frame.joints[joint2].y_normalized
                )
                / 2
            )
            assert (
                avg_joint.z_normalized
                == (
                    frame.joints[joint1].z_normalized
                    + frame.joints[joint2].z_normalized
                )
                / 2
            )
            assert avg_joint.image_dimensions == frame.image_dimensions


class TestBlazePoseFrameMeasurements:
    """Tests for BlazePoseFrame measurement-related methods."""

    def test_generate_angle_measurements_no_joints(self, sample_frame_data):
        """
        GIVEN a frame without joint positions
        WHEN generate_angle_measurements is called
        THEN BlazePoseFrameError is raised
        """
        # Arrange
        frame = BlazePoseFrame(frame_data=sample_frame_data)

        # Act & Assert
        with pytest.raises(
            BlazePoseFrameError, match="There are no joint data to generate angles from"
        ):
            frame.generate_angle_measurements({})

    def test_generate_angle_measurements(
        self, sample_frame_data_with_joints, mock_transformer
    ):
        """
        GIVEN a frame with joint positions and vectors
        WHEN generate_angle_measurements is called
        THEN Angle objects are created according to the map
        """
        # Arrange
        frame = BlazePoseFrame(frame_data=sample_frame_data_with_joints)

        # Mock the OpenPoseMediapipeTransformer and setup vectors
        mock_transformer.create_openpose_joints_and_vectors.return_value = True
        frame.has_openpose_joints_and_vectors = True

        # Setup test vectors
        vector1_name = "vector1"
        vector2_name = "vector2"
        frame.vectors = {
            vector1_name: MagicMock(spec=Vector),
            vector2_name: MagicMock(spec=Vector),
        }

        # Prepare mock angle map
        angle_map = {"test_angle": (vector1_name, vector2_name)}

        # Mock the Angle constructor
        with patch("stream_pose_ml.blaze_pose.blaze_pose_frame.Angle") as mock_angle:
            mock_angle.return_value = MagicMock(spec=Angle)

            # Act
            angles = frame.generate_angle_measurements(angle_map)

            # Assert
            assert "test_angle" in angles
            mock_angle.assert_called_once_with(
                "test_angle", frame.vectors[vector1_name], frame.vectors[vector2_name]
            )

    def test_generate_distance_measurements_no_joints(self, sample_frame_data):
        """
        GIVEN a frame without joint positions
        WHEN generate_distance_measurements is called
        THEN BlazePoseFrameError is raised
        """
        # Arrange
        frame = BlazePoseFrame(frame_data=sample_frame_data)

        # Act & Assert
        with pytest.raises(
            BlazePoseFrameError,
            match="There are no joint data to generate distances from",
        ):
            frame.generate_distance_measurements({})

    def test_generate_distance_measurements(
        self, sample_frame_data_with_joints, mock_transformer
    ):
        """
        GIVEN a frame with joint positions and vectors
        WHEN generate_distance_measurements is called
        THEN Distance objects are created according to the map
        """
        # Arrange
        frame = BlazePoseFrame(frame_data=sample_frame_data_with_joints)

        # Mock the OpenPoseMediapipeTransformer and setup vectors and joints
        mock_transformer.create_openpose_joints_and_vectors.return_value = True
        frame.has_openpose_joints_and_vectors = True

        # Setup test vectors and joints
        joint_name = "joint1"
        vector_name = "vector1"
        frame.vectors = {vector_name: MagicMock(spec=Vector)}
        frame.joints = {joint_name: MagicMock(spec=Joint)}

        # Prepare mock distance map
        distance_map = {"test_distance": (joint_name, vector_name)}

        # Mock the Distance constructor
        with patch(
            "stream_pose_ml.blaze_pose.blaze_pose_frame.Distance"
        ) as mock_distance:
            mock_distance.return_value = MagicMock(spec=Distance)

            # Act
            distances = frame.generate_distance_measurements(distance_map)

            # Assert
            assert "test_distance" in distances
            mock_distance.assert_called_once_with(
                "test_distance", frame.joints[joint_name], frame.vectors[vector_name]
            )


class TestBlazePoseFrameIntegration:
    """Integration tests using real examples to verify end-to-end behavior."""

    @pytest.fixture
    def example_data_path(self):
        """Returns the path to example data used for integration tests."""
        base_path = Path(__file__).parent.parent
        return base_path / "example_data"

    @pytest.fixture
    def blaze_pose_sequence(self, example_data_path):
        """Create a real BlazePoseSequence using example data."""
        from stream_pose_ml.blaze_pose.blaze_pose_sequence import BlazePoseSequence
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

        bps = BlazePoseSequence(name="test", sequence=mpc.frame_data_list)

        yield bps

        # Cleanup
        import shutil

        try:
            shutil.rmtree(output_path)
        except OSError:
            pass

    def test_full_frame_initialization_with_real_data(self, blaze_pose_sequence):
        """
        GIVEN real data from BlazePoseSequence
        WHEN BlazePoseFrame is created with real data
        THEN it correctly processes the frame with all features
        """
        # Skip if no sequence data available
        if not blaze_pose_sequence.sequence_data:
            pytest.skip("No sequence data available")

        # Get frame data from sequence
        frame_data = blaze_pose_sequence.sequence_data[0]

        # Create frame with all options enabled
        frame = BlazePoseFrame(
            frame_data=frame_data, generate_angles=True, generate_distances=True
        )

        # Verify basic properties
        assert frame.frame_number == frame_data["frame_number"]
        assert frame.sequence_id == frame_data["sequence_id"]
        assert frame.sequence_source == frame_data["sequence_source"]
        assert frame.image_dimensions == frame_data["image_dimensions"]

        # Verify joint positions are processed
        if frame.has_joint_positions:
            assert len(frame.joints) > 0
            assert all(isinstance(joint, Joint) for joint in frame.joints.values())

            # Verify angles and distances
            assert frame.has_openpose_joints_and_vectors is True
            assert len(frame.angles) > 0
            assert len(frame.distances) > 0
            assert all(isinstance(angle, Angle) for angle in frame.angles.values())
            assert all(
                isinstance(distance, Distance) for distance in frame.distances.values()
            )

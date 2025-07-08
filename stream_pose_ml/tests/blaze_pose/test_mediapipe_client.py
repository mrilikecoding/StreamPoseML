"""Tests for the MediaPipeClient class."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add the project root to the Python path
project_root = Path(__file__).parents[
    3
]  # /Users/nathangreen/Development/stream_pose_ml
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ruff: noqa: E402
from stream_pose_ml.blaze_pose.enumerations import BlazePoseJoints
from stream_pose_ml.blaze_pose.mediapipe_client import (
    MediaPipeClient,
    MediaPipeClientError,
)


@pytest.fixture
def mock_cv2():
    """Create a mock for cv2."""
    with patch("stream_pose_ml.blaze_pose.mediapipe_client.cv2") as mock:
        # Mock VideoCapture and its methods
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (True, np.zeros((720, 1280, 3))),  # First frame
            (True, np.zeros((720, 1280, 3))),  # Second frame
            (False, None),  # End of video
        ]
        mock.VideoCapture.return_value = mock_cap

        # Mock color conversion
        mock.cvtColor.return_value = np.zeros((720, 1280, 3))

        # Mock CLAHE for preprocessing
        mock_clahe = MagicMock()
        mock_clahe.apply.return_value = np.zeros((720, 1080))
        mock.createCLAHE.return_value = mock_clahe

        # Mock color space conversion functions
        mock.COLOR_BGR2RGB = 4
        mock.COLOR_BGR2LAB = 44
        mock.COLOR_LAB2BGR = 55

        # Mock split and merge
        mock.split.return_value = (
            np.zeros((720, 1080)),
            np.zeros((720, 1080)),
            np.zeros((720, 1080)),
        )
        mock.merge.return_value = np.zeros((720, 1080, 3))

        yield mock


@pytest.fixture
def mock_mediapipe():
    """Create a mock for mediapipe."""
    with patch("stream_pose_ml.blaze_pose.mediapipe_client.mp") as mock:
        # Mock pose solution
        mock_pose = MagicMock()
        mock.solutions.pose.Pose.return_value = mock_pose

        # Mock pose results
        mock_landmark = MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.0

        mock_pose_landmarks = MagicMock()
        mock_pose_landmarks.landmark = [
            mock_landmark
        ] * 33  # 33 landmarks for BlazePose

        mock_results = MagicMock()
        mock_results.pose_landmarks = mock_pose_landmarks

        # Set up process to return our mock results
        mock_pose.process.return_value = mock_results

        yield mock


@pytest.fixture
def mock_os():
    """Create a mock for os."""
    with patch("stream_pose_ml.blaze_pose.mediapipe_client.os") as mock:
        mock.makedirs.return_value = None
        yield mock


@pytest.fixture
def example_data_path():
    """Returns the path to the example data directory."""
    return Path(__file__).parent.parent / "example_data"


class TestMediaPipeClientInitialization:
    """Tests for MediaPipeClient initialization."""

    def test_init_with_valid_parameters(self):
        """
        GIVEN valid initialization parameters
        WHEN MediaPipeClient is initialized
        THEN the client is created with the correct attributes
        """
        # Arrange & Act
        client = MediaPipeClient(
            video_input_filename="test.mp4",
            video_input_path="/test/path",
            video_output_prefix="/output/path",
            id=12345,
            configuration={"test": "config"},
            preprocess_video=True,
        )

        # Assert
        assert client.video_input_filename == "test.mp4"
        assert client.video_input_path == "/test/path"
        assert client.video_output_prefix == "/output/path"
        assert client.id == 12345
        assert client.configuration == {"test": "config"}
        assert client.preprocess_video is True
        assert client.frame_count == 0
        assert client.frame_data_list == []
        assert client.json_output_path == "/output/path/test-12345"

    def test_init_without_input_filename(self):
        """
        GIVEN initialization without a video input filename
        WHEN MediaPipeClient is initialized
        THEN MediaPipeClientError is raised
        """
        # Act & Assert
        with pytest.raises(MediaPipeClientError, match="No input file specified"):
            MediaPipeClient(
                video_input_path="/test/path", video_output_prefix="/output/path"
            )

    def test_init_with_dummy_client(self):
        """
        GIVEN dummy_client=True
        WHEN MediaPipeClient is initialized without a filename
        THEN no error is raised
        """
        # Act
        client = MediaPipeClient(dummy_client=True)

        # Assert
        assert client.video_input_filename is None

    def test_init_sets_joint_list(self):
        """
        GIVEN a MediaPipeClient
        WHEN it is initialized
        THEN joints list is populated from BlazePoseJoints enum
        """
        # Act
        client = MediaPipeClient(video_input_filename="test.mp4", dummy_client=False)

        # Assert
        assert len(client.joints) > 0
        assert "nose" in client.joints
        assert "left_shoulder" in client.joints
        assert client.joints == [joint.name for joint in BlazePoseJoints]


class TestMediaPipeClientProcessing:
    """Tests for MediaPipeClient video processing methods."""

    def test_process_video(self, mock_cv2, mock_mediapipe):
        """
        GIVEN a MediaPipeClient with mocked dependencies
        WHEN process_video is called
        THEN video frames are processed and frame data is collected
        """
        # Arrange
        client = MediaPipeClient(
            video_input_filename="test.mp4",
            video_input_path="/test/path",
            video_output_prefix="/output/path",
        )

        # Replace the serialize_pose_landmarks method with a mock
        client.serialize_pose_landmarks = MagicMock(
            return_value={
                "nose": {
                    "x": 0.5,
                    "y": 0.5,
                    "z": 0.0,
                    "x_normalized": 0.0,
                    "y_normalized": 0.0,
                    "z_normalized": 0.0,
                }
            }
        )

        # Act
        result = client.process_video()

        # Assert
        assert result is client  # Returns self for chaining
        assert client.frame_count == 2  # Processed 2 frames before end of video
        assert len(client.frame_data_list) == 2
        assert mock_cv2.VideoCapture.called
        assert mock_mediapipe.solutions.pose.Pose.called

        # Check frame data structure
        for frame_data in client.frame_data_list:
            assert "sequence_id" in frame_data
            assert "sequence_source" in frame_data
            assert "frame_number" in frame_data
            assert "image_dimensions" in frame_data
            assert "joint_positions" in frame_data

    def test_process_video_with_limit(self, mock_cv2, mock_mediapipe):
        """
        GIVEN a MediaPipeClient with mocked dependencies
        WHEN process_video is called with a limit
        THEN only specified number of frames are processed
        """
        # Arrange
        client = MediaPipeClient(
            video_input_filename="test.mp4",
            video_input_path="/test/path",
            video_output_prefix="/output/path",
        )

        # Replace the serialize_pose_landmarks method with a mock
        client.serialize_pose_landmarks = MagicMock(
            return_value={
                "nose": {
                    "x": 0.5,
                    "y": 0.5,
                    "z": 0.0,
                    "x_normalized": 0.0,
                    "y_normalized": 0.0,
                    "z_normalized": 0.0,
                }
            }
        )

        # Act
        client.process_video(limit=1)

        # Assert
        assert client.frame_count == 1  # Only processed 1 frame due to limit
        assert len(client.frame_data_list) == 1

    def test_process_video_file_not_found(self, mock_cv2):
        """
        GIVEN a MediaPipeClient with a non-existent video file
        WHEN process_video is called
        THEN MediaPipeClientError is raised
        """
        # Arrange
        client = MediaPipeClient(
            video_input_filename="nonexistent.mp4",
            video_input_path="/test/path",
            video_output_prefix="/output/path",
        )

        # Mock VideoCapture for this specific file to return isOpened=False
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap

        # Act & Assert
        with pytest.raises(MediaPipeClientError, match="Error opening file"):
            client.process_video()

    def test_process_video_with_preprocess(self, mock_cv2, mock_mediapipe):
        """
        GIVEN a MediaPipeClient with preprocess_video=True
        WHEN process_video is called
        THEN frames are preprocessed before pose detection
        """
        # Arrange
        client = MediaPipeClient(
            video_input_filename="test.mp4",
            video_input_path="/test/path",
            video_output_prefix="/output/path",
            preprocess_video=True,
        )

        # Mock the preprocessing method
        with patch.object(
            MediaPipeClient,
            "run_preprocess_video",
            return_value=np.zeros((720, 1280, 3)),
        ) as mock_preprocess:
            # Mock the serialize_pose_landmarks method
            client.serialize_pose_landmarks = MagicMock(return_value={})

            # Act
            client.process_video(limit=1)

            # Assert
            assert mock_preprocess.called

    def test_process_video_no_pose_detected(self, mock_cv2, mock_mediapipe):
        """
        GIVEN a MediaPipeClient where pose detection fails
        WHEN process_video is called
        THEN frame data still contains metadata but empty joint positions
        """
        # Arrange
        client = MediaPipeClient(
            video_input_filename="test.mp4",
            video_input_path="/test/path",
            video_output_prefix="/output/path",
        )

        # Change the mock to return no pose landmarks
        mock_results = MagicMock()
        mock_results.pose_landmarks = None
        mock_mediapipe.solutions.pose.Pose().process.return_value = mock_results

        # Act
        client.process_video(limit=1)

        # Assert
        assert len(client.frame_data_list) == 1
        assert client.frame_data_list[0]["joint_positions"] == {}


class TestMediaPipeClientPoseLandmarks:
    """Tests for MediaPipeClient pose landmark processing."""

    def test_serialize_pose_landmarks(self):
        """
        GIVEN a list of pose landmarks
        WHEN serialize_pose_landmarks is called
        THEN landmarks are serialized into the expected format
        """
        # Arrange
        client = MediaPipeClient(video_input_filename="test.mp4", dummy_client=False)

        # Create mock landmarks
        class MockLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        landmarks = [MockLandmark(0.1, 0.2, 0.3)] * len(client.joints)

        # Mock the joint coordinate calculation methods
        with patch.object(MediaPipeClient, "get_joint_coordinates") as mock_get_coords:
            with patch.object(
                MediaPipeClient, "calculate_reference_point_distance"
            ) as mock_calc_dist:
                with patch.object(
                    MediaPipeClient, "calculate_reference_point_midpoint"
                ) as mock_calc_mid:
                    # Set up the mock return values
                    mock_get_coords.return_value = [0.5, 0.5]
                    mock_calc_dist.return_value = 1.0
                    mock_calc_mid.return_value = {"x": 0.5, "y": 0.5}

                    # Act
                    result = client.serialize_pose_landmarks(landmarks)

                    # Assert
                    assert isinstance(result, dict)
                    assert len(result) == len(client.joints)

                    # Check structure of a serialized joint
                    for _joint_name, joint_data in result.items():
                        assert "x" in joint_data
                        assert "y" in joint_data
                        assert "z" in joint_data
                        assert "x_normalized" in joint_data
                        assert "y_normalized" in joint_data
                        assert "z_normalized" in joint_data

                    # Verify the calculation methods were called
                    assert mock_get_coords.call_count >= 2
                    assert mock_calc_dist.called
                    assert mock_calc_mid.called

    def test_get_joint_coordinates_from_landmark_object(self):
        """
        GIVEN a list of landmark objects
        WHEN get_joint_coordinates is called
        THEN coordinates are extracted correctly
        """
        # Arrange
        MediaPipeClient(dummy_client=True)

        # Create a mock landmark
        class MockLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        landmarks = [
            MockLandmark(0.1, 0.2),
            MockLandmark(0.3, 0.4),
            MockLandmark(0.5, 0.6),
        ]

        # Act - get coordinates of the second landmark
        result = MediaPipeClient.get_joint_coordinates(
            joints=["joint1", "joint2", "joint3"],
            reference_joint_name="joint2",
            pose_landmarks=landmarks,
        )

        # Assert
        assert result == [0.3, 0.4]

    def test_get_joint_coordinates_from_dictionary(self):
        """
        GIVEN a list of landmark dictionaries
        WHEN get_joint_coordinates is called
        THEN coordinates are extracted correctly
        """
        # Arrange
        MediaPipeClient(dummy_client=True)

        # Create a mock landmark dictionary
        landmarks = [{"x": 0.1, "y": 0.2}, {"x": 0.3, "y": 0.4}, {"x": 0.5, "y": 0.6}]

        # Act - get coordinates of the second landmark
        result = MediaPipeClient.get_joint_coordinates(
            joints=["joint1", "joint2", "joint3"],
            reference_joint_name="joint2",
            pose_landmarks=landmarks,
        )

        # Assert
        assert result == [0.3, 0.4]

    def test_calculate_reference_point_distance(self):
        """
        GIVEN two joint coordinates
        WHEN calculate_reference_point_distance is called
        THEN the Euclidean distance is calculated correctly
        """
        # Arrange
        joint1 = [0.0, 0.0]
        joint2 = [3.0, 4.0]

        # Act
        result = MediaPipeClient.calculate_reference_point_distance(joint1, joint2)

        # Assert
        assert result == 5.0  # 3-4-5 triangle

    def test_calculate_reference_point_midpoint(self):
        """
        GIVEN two joint coordinates
        WHEN calculate_reference_point_midpoint is called
        THEN the midpoint is calculated correctly
        """
        # Arrange
        joint1 = [1.0, 2.0]
        joint2 = [3.0, 4.0]

        # Act
        result = MediaPipeClient.calculate_reference_point_midpoint(joint1, joint2)

        # Assert
        assert result == {"x": 2.0, "y": 3.0}


class TestMediaPipeClientPreprocessing:
    """Tests for MediaPipeClient video preprocessing."""

    def test_run_preprocess_video(self, mock_cv2):
        """
        GIVEN an image
        WHEN run_preprocess_video is called
        THEN the image is preprocessed with contrast enhancement
        """
        # Arrange
        image = np.zeros((720, 1280, 3))

        # Act
        result = MediaPipeClient.run_preprocess_video(image)

        # Assert
        assert mock_cv2.cvtColor.call_count >= 2  # At least two color space conversions
        assert mock_cv2.createCLAHE.called
        assert mock_cv2.split.called
        assert mock_cv2.merge.called
        assert isinstance(result, np.ndarray)
        assert result.shape == image.shape


class TestMediaPipeClientFileOutput:
    """Tests for MediaPipeClient file output methods."""

    def test_write_pose_data_to_file(self, mock_os):
        """
        GIVEN a MediaPipeClient with frame data
        WHEN write_pose_data_to_file is called
        THEN frame data is written to JSON files
        """
        # Arrange
        client = MediaPipeClient(
            video_input_filename="test.mp4",
            video_input_path="/test/path",
            video_output_prefix="/output/path",
        )

        # Add some frame data
        client.frame_data_list = [
            {
                "sequence_id": client.id,
                "sequence_source": "mediapipe",
                "frame_number": 1,
                "image_dimensions": {"height": 720, "width": 1280},
                "joint_positions": {},
            },
            {
                "sequence_id": client.id,
                "sequence_source": "mediapipe",
                "frame_number": 2,
                "image_dimensions": {"height": 720, "width": 1280},
                "joint_positions": {},
            },
        ]

        # Mock the file operations
        mock_file = MagicMock()
        open_mock = MagicMock(return_value=mock_file)

        # Act
        with patch("builtins.open", open_mock):
            with patch("json.dump") as mock_json_dump:
                client.write_pose_data_to_file()

                # Assert
                assert mock_os.makedirs.called
                assert mock_os.makedirs.call_args[0][0] == client.json_output_path
                assert open_mock.call_count == 2  # Two files should be opened
                assert mock_json_dump.call_count == 2  # Two JSON dumps should happen

    def test_write_pose_data_to_file_error(self, mock_os):
        """
        GIVEN a MediaPipeClient where file writing fails
        WHEN write_pose_data_to_file is called
        THEN MediaPipeClientError is raised
        """
        # Arrange
        client = MediaPipeClient(
            video_input_filename="test.mp4",
            video_input_path="/test/path",
            video_output_prefix="/output/path",
        )

        # Make os.makedirs raise an exception
        mock_os.makedirs.side_effect = Exception("Test error")

        # Act & Assert
        with pytest.raises(
            MediaPipeClientError, match="There was a problem writing pose data to json"
        ):
            client.write_pose_data_to_file()

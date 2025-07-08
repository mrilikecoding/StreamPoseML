import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[
    3
]  # /Users/nathangreen/Development/stream_pose_ml
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from unittest.mock import MagicMock, patch

import pytest

from stream_pose_ml.services.video_data_service import (
    VideoDataService,
    VideoDataServiceError,
)


class TestVideoDataService:
    """Test the VideoDataService class."""

    @pytest.fixture
    def mock_time(self):
        """Create a mock for time."""
        with patch("stream_pose_ml.services.video_data_service.time") as mock_time:
            mock_time.time_ns.return_value = 123456789
            yield mock_time

    @pytest.fixture
    def mock_media_pipe_client(self):
        """Create a mock for MediaPipeClient."""
        with patch(
            "stream_pose_ml.services.video_data_service.MediaPipeClient"
        ) as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.process_video.return_value = mock_instance

            # Set up mock frame data
            mock_instance.frame_data_list = [
                {"frame_number": 1, "keypoints": {"test": "data1"}},
                {"frame_number": 2, "keypoints": {"test": "data2"}},
            ]

            yield mock_client

    @pytest.fixture
    def mock_blaze_pose_sequence(self):
        """Create a mock for BlazePoseSequence."""
        with patch(
            "stream_pose_ml.services.video_data_service.BlazePoseSequence"
        ) as mock_sequence:
            mock_instance = MagicMock()
            mock_sequence.return_value = mock_instance
            mock_instance.generate_blaze_pose_frames_from_sequence.return_value = (
                mock_instance
            )

            yield mock_sequence

    @pytest.fixture
    def mock_serializer(self):
        """Create a mock for BlazePoseSequenceSerializer."""
        with patch(
            "stream_pose_ml.services.video_data_service.BlazePoseSequenceSerializer"
        ) as mock_serializer:
            mock_instance = MagicMock()
            mock_serializer.return_value = mock_instance

            # Serializer returns test data
            mock_instance.serialize.return_value = {
                "name": "test_video.mp4",
                "frames": {
                    "1": {"frame_number": 1, "data": "test1"},
                    "2": {"frame_number": 2, "data": "test2"},
                },
            }

            yield mock_serializer

    @pytest.fixture
    def mock_path_utility(self):
        """Create a mock for path_utility."""
        with patch(
            "stream_pose_ml.services.video_data_service.path_utility"
        ) as mock_utility:
            mock_utility.write_to_json_file.return_value = True
            yield mock_utility

    def test_process_video_success(
        self,
        mock_time,
        mock_media_pipe_client,
        mock_blaze_pose_sequence,
        mock_serializer,
        mock_path_utility,
    ):
        """Test the basic video processing flow."""
        # Given
        service = VideoDataService()
        input_filename = "test_video.mp4"
        video_input_path = "/path/to/videos"

        # When
        result = service.process_video(
            input_filename=input_filename, video_input_path=video_input_path
        )

        # Then
        assert result == {
            "name": "test_video.mp4",
            "frames": {
                "1": {"frame_number": 1, "data": "test1"},
                "2": {"frame_number": 2, "data": "test2"},
            },
        }

        # Verify MediaPipeClient was created and called correctly
        mock_media_pipe_client.assert_called_once_with(
            video_input_filename=input_filename,
            video_input_path=video_input_path,
            video_output_prefix="",
            id=123456789,
            configuration={},
            preprocess_video=False,
        )
        mock_media_pipe_client.return_value.process_video.assert_called_once()

        # Verify BlazePoseSequence was created and called correctly
        mock_blaze_pose_sequence.assert_called_once_with(
            name=input_filename,
            sequence=mock_media_pipe_client.return_value.frame_data_list,
            include_geometry=True,
        )
        mock_blaze_pose_sequence.return_value.generate_blaze_pose_frames_from_sequence.assert_called_once()

        # Verify serializer was called
        mock_serializer.return_value.serialize.assert_called_once_with(
            mock_blaze_pose_sequence.return_value.generate_blaze_pose_frames_from_sequence.return_value,
            key_off_frame_number=True,
        )

        # File should not be written
        mock_path_utility.write_to_json_file.assert_not_called()

    def test_process_video_write_keypoints(
        self,
        mock_media_pipe_client,
        mock_blaze_pose_sequence,
        mock_serializer,
        mock_path_utility,
    ):
        """Test writing keypoints to file."""
        # Given
        service = VideoDataService()
        input_filename = "test_video.mp4"
        video_input_path = "/path/to/videos"
        output_keypoint_data_path = "/path/to/keypoints"

        # When
        service.process_video(
            input_filename=input_filename,
            video_input_path=video_input_path,
            write_keypoints_to_file=True,
            output_keypoint_data_path=output_keypoint_data_path,
        )

        # Then
        # Verify MediaPipeClient was created with correct output prefix
        mock_media_pipe_client.assert_called_once_with(
            video_input_filename=input_filename,
            video_input_path=video_input_path,
            video_output_prefix=output_keypoint_data_path,
            id=mock_media_pipe_client.call_args[1]["id"],  # Use dynamic ID
            configuration={},
            preprocess_video=False,
        )

        # Verify keypoints were written
        mock_media_pipe_client.return_value.write_pose_data_to_file.assert_called_once()

    def test_process_video_write_sequence(
        self,
        mock_media_pipe_client,
        mock_blaze_pose_sequence,
        mock_serializer,
        mock_path_utility,
    ):
        """Test writing sequence data to file."""
        # Given
        service = VideoDataService()
        input_filename = "test_video.mp4"
        video_input_path = "/path/to/videos"
        output_sequence_data_path = "/path/to/sequences"

        # When
        service.process_video(
            input_filename=input_filename,
            video_input_path=video_input_path,
            write_serialized_sequence_to_file=True,
            output_sequence_data_path=output_sequence_data_path,
        )

        # Then
        # Verify sequence data was written
        mock_path_utility.write_to_json_file.assert_called_once_with(
            output_sequence_data_path,
            f"{input_filename}_sequence.json",
            mock_serializer.return_value.serialize.return_value,
        )

    def test_process_video_no_keypoint_path_error(self, mock_media_pipe_client):
        """Test error when write_keypoints_to_file is True but no path is provided."""
        # Given
        service = VideoDataService()
        input_filename = "test_video.mp4"
        video_input_path = "/path/to/videos"

        # When/Then
        with pytest.raises(
            VideoDataServiceError, match="No output path specified for keypoint data."
        ):
            service.process_video(
                input_filename=input_filename,
                video_input_path=video_input_path,
                write_keypoints_to_file=True,
                output_keypoint_data_path=None,
            )

    def test_process_video_no_sequence_path_error(
        self, mock_media_pipe_client, mock_blaze_pose_sequence, mock_serializer
    ):
        """Test error when write_serialized_sequence_to_file is True but no path is
        provided."""
        # Given
        service = VideoDataService()
        input_filename = "test_video.mp4"
        video_input_path = "/path/to/videos"

        # When/Then
        with pytest.raises(
            VideoDataServiceError,
            match="No output path specified for serialzied sequence data.",
        ):
            service.process_video(
                input_filename=input_filename,
                video_input_path=video_input_path,
                write_serialized_sequence_to_file=True,
                output_sequence_data_path=None,
            )

    def test_process_video_with_custom_config(
        self, mock_media_pipe_client, mock_blaze_pose_sequence, mock_serializer
    ):
        """Test processing a video with custom configuration."""
        # Given
        service = VideoDataService()
        input_filename = "test_video.mp4"
        video_input_path = "/path/to/videos"
        custom_config = {"min_detection_confidence": 0.8, "model_complexity": 2}

        # When
        service.process_video(
            input_filename=input_filename,
            video_input_path=video_input_path,
            configuration=custom_config,
        )

        # Then
        # Verify MediaPipeClient was created with custom config
        mock_media_pipe_client.assert_called_once_with(
            video_input_filename=input_filename,
            video_input_path=video_input_path,
            video_output_prefix="",
            id=mock_media_pipe_client.call_args[1]["id"],  # Use dynamic ID
            configuration=custom_config,
            preprocess_video=False,
        )

    def test_process_video_no_geometry(
        self, mock_media_pipe_client, mock_blaze_pose_sequence, mock_serializer
    ):
        """Test processing a video without geometry computations."""
        # Given
        service = VideoDataService()
        input_filename = "test_video.mp4"
        video_input_path = "/path/to/videos"

        # When
        service.process_video(
            input_filename=input_filename,
            video_input_path=video_input_path,
            include_geometry=False,
        )

        # Then
        # Verify BlazePoseSequence was created with include_geometry=False
        mock_blaze_pose_sequence.assert_called_once_with(
            name=input_filename,
            sequence=mock_media_pipe_client.return_value.frame_data_list,
            include_geometry=False,
        )

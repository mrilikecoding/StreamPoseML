import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parents[3]  # Adjust if needed
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ruff: noqa: E402
from stream_pose_ml.jobs.process_video_job import ProcessVideoJob, ProcessVideoJobError


class TestProcessVideoJob:
    """Test the ProcessVideoJob class."""

    @pytest.fixture
    def mock_video_data_service(self):
        """Create a mock for VideoDataService."""
        with patch(
            "stream_pose_ml.jobs.process_video_job.VideoDataService"
        ) as mock_service:
            mock_instance = MagicMock()
            mock_service.return_value = mock_instance
            mock_instance.process_video.return_value = {"mock": "data"}
            yield mock_service

    def test_process_video_success(self, mock_video_data_service):
        """Test processing a video successfully."""
        # Given
        input_filename = "test_video.mp4"
        video_input_path = "/path/to/videos"
        output_keypoint_data_path = "/path/to/keypoints"
        output_sequence_data_path = "/path/to/sequences"

        # When
        result = ProcessVideoJob.process_video(
            input_filename=input_filename,
            video_input_path=video_input_path,
            output_keypoint_data_path=output_keypoint_data_path,
            output_sequence_data_path=output_sequence_data_path,
            write_keypoints_to_file=True,
            write_serialized_sequence_to_file=True,
            configuration={"mock": "config"},
            preprocess_video=True,
        )

        # Then
        assert result == {"mock": "data"}
        mock_video_data_service.return_value.process_video.assert_called_once_with(
            input_filename=input_filename,
            video_input_path=video_input_path,
            output_keypoint_data_path=output_keypoint_data_path,
            output_sequence_data_path=output_sequence_data_path,
            write_keypoints_to_file=True,
            write_serialized_sequence_to_file=True,
            configuration={"mock": "config"},
            preprocess_video=True,
        )

    def test_process_video_no_keypoint_path(self, mock_video_data_service):
        """Test error when write_keypoints_to_file is True but no path is provided."""
        # Given
        input_filename = "test_video.mp4"
        video_input_path = "/path/to/videos"

        # When/Then
        with pytest.raises(
            ProcessVideoJobError,
            match="No output location specified for keypoints files.",
        ):
            ProcessVideoJob.process_video(
                input_filename=input_filename,
                video_input_path=video_input_path,
                output_keypoint_data_path=None,
                output_sequence_data_path="/path/to/sequences",
                write_keypoints_to_file=True,
                write_serialized_sequence_to_file=False,
            )

        # Verify the service was not called
        mock_video_data_service.return_value.process_video.assert_not_called()

    def test_process_video_no_sequence_path(self, mock_video_data_service):
        """Test error when write_serialized_sequence_to_file is True but no path is
        provided."""
        # Given
        input_filename = "test_video.mp4"
        video_input_path = "/path/to/videos"

        # When/Then
        with pytest.raises(
            ProcessVideoJobError,
            match="No output location specified for sequence data files.",
        ):
            ProcessVideoJob.process_video(
                input_filename=input_filename,
                video_input_path=video_input_path,
                output_keypoint_data_path="/path/to/keypoints",
                output_sequence_data_path=None,
                write_keypoints_to_file=False,
                write_serialized_sequence_to_file=True,
            )

        # Verify the service was not called
        mock_video_data_service.return_value.process_video.assert_not_called()

    def test_process_video_no_file_writing(self, mock_video_data_service):
        """Test processing a video without writing files."""
        # Given
        input_filename = "test_video.mp4"
        video_input_path = "/path/to/videos"

        # When
        result = ProcessVideoJob.process_video(
            input_filename=input_filename,
            video_input_path=video_input_path,
            output_keypoint_data_path=None,
            output_sequence_data_path=None,
            write_keypoints_to_file=False,
            write_serialized_sequence_to_file=False,
        )

        # Then
        assert result == {"mock": "data"}
        mock_video_data_service.return_value.process_video.assert_called_once_with(
            input_filename=input_filename,
            video_input_path=video_input_path,
            output_keypoint_data_path=None,
            output_sequence_data_path=None,
            write_keypoints_to_file=False,
            write_serialized_sequence_to_file=False,
            configuration={},
            preprocess_video=False,
        )

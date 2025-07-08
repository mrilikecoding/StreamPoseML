import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parents[3]  # Adjust if needed
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ruff: noqa: E402
from stream_pose_ml.jobs.process_videos_job import ProcessVideosJob


class TestProcessVideosJob:
    """Test the ProcessVideosJob class."""

    @pytest.fixture
    def mock_path_utility(self):
        """Create a mock for path_utility functions."""
        with patch(
            "stream_pose_ml.jobs.process_videos_job.path_utility"
        ) as mock_utility:
            mock_utility.get_file_paths_in_directory.side_effect = [
                ["/path/to/video1.webm", "/path/to/video2.webm"],  # webm files
                [],  # mp4 files (none in this test)
            ]
            mock_utility.get_file_name.side_effect = lambda path: path.split("/")[-1]
            mock_utility.get_base_path.side_effect = lambda path: "/".join(
                path.split("/")[:-1]
            )
            yield mock_utility

    @pytest.fixture
    def mock_process_video_job(self):
        """Create a mock for ProcessVideoJob."""
        with patch(
            "stream_pose_ml.jobs.process_videos_job.ProcessVideoJob"
        ) as mock_job:
            mock_job.process_video.side_effect = [
                {"video1": "data"},
                {"video2": "data"},
            ]
            yield mock_job

    def test_process_videos_with_return_output(
        self, mock_path_utility, mock_process_video_job
    ):
        """Test processing videos with return_output=True."""
        # Given
        src_videos_path = "/path/to/videos"
        output_keypoints_data_path = "/path/to/keypoints"
        output_sequence_data_path = "/path/to/sequences"
        configuration = {"mock": "config"}

        # When
        result = ProcessVideosJob.process_videos(
            src_videos_path=src_videos_path,
            output_keypoints_data_path=output_keypoints_data_path,
            output_sequence_data_path=output_sequence_data_path,
            write_keypoints_to_file=True,
            write_serialized_sequence_to_file=True,
            configuration=configuration,
            return_output=True,
        )

        # Then
        assert result == [{"video1": "data"}, {"video2": "data"}]
        assert mock_process_video_job.process_video.call_count == 2

        # Verify first call
        mock_process_video_job.process_video.assert_any_call(
            input_filename="video1.webm",
            video_input_path="/path/to",
            output_keypoint_data_path=output_keypoints_data_path,
            output_sequence_data_path=output_sequence_data_path,
            write_keypoints_to_file=True,
            write_serialized_sequence_to_file=True,
            configuration=configuration,
            preprocess_video=False,
        )

        # Verify second call
        mock_process_video_job.process_video.assert_any_call(
            input_filename="video2.webm",
            video_input_path="/path/to",
            output_keypoint_data_path=output_keypoints_data_path,
            output_sequence_data_path=output_sequence_data_path,
            write_keypoints_to_file=True,
            write_serialized_sequence_to_file=True,
            configuration=configuration,
            preprocess_video=False,
        )

    def test_process_videos_without_return_output(
        self, mock_path_utility, mock_process_video_job
    ):
        """Test processing videos with return_output=False."""
        # Given
        src_videos_path = "/path/to/videos"
        output_keypoints_data_path = "/path/to/keypoints"
        output_sequence_data_path = "/path/to/sequences"

        # When
        result = ProcessVideosJob.process_videos(
            src_videos_path=src_videos_path,
            output_keypoints_data_path=output_keypoints_data_path,
            output_sequence_data_path=output_sequence_data_path,
            return_output=False,
        )

        # Then
        assert result == {
            "keypoints_path": output_keypoints_data_path,
            "sequence_path": output_sequence_data_path,
        }
        assert mock_process_video_job.process_video.call_count == 2

    def test_process_videos_with_limit(self, mock_path_utility, mock_process_video_job):
        """Test processing videos with a limit."""
        # Given
        src_videos_path = "/path/to/videos"
        limit = 1

        # When
        result = ProcessVideosJob.process_videos(
            src_videos_path=src_videos_path, limit=limit
        )

        # Then
        assert result == [{"video1": "data"}]  # Only first video processed due to limit
        assert mock_process_video_job.process_video.call_count == 1

    def test_process_videos_file_extensions(
        self, mock_path_utility, mock_process_video_job
    ):
        """Test processing videos handles different file extensions."""
        # Modify mock to return both webm and mp4 files
        mock_path_utility.get_file_paths_in_directory.side_effect = [
            ["/path/to/video1.webm"],  # webm files
            ["/path/to/video2.mp4"],  # mp4 files
        ]

        # When
        ProcessVideosJob.process_videos(src_videos_path="/path/to/videos")

        # Then
        assert mock_process_video_job.process_video.call_count == 2

        # Verify both file types were processed
        mock_process_video_job.process_video.assert_any_call(
            input_filename="video1.webm",
            video_input_path="/path/to",
            output_keypoint_data_path="",
            output_sequence_data_path="",
            write_keypoints_to_file=False,
            write_serialized_sequence_to_file=False,
            configuration={},
            preprocess_video=False,
        )

        mock_process_video_job.process_video.assert_any_call(
            input_filename="video2.mp4",
            video_input_path="/path/to",
            output_keypoint_data_path="",
            output_sequence_data_path="",
            write_keypoints_to_file=False,
            write_serialized_sequence_to_file=False,
            configuration={},
            preprocess_video=False,
        )

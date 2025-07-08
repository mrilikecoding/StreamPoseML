import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[
    3
]  # /Users/nathangreen/Development/stream_pose_ml
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from unittest.mock import MagicMock, mock_open, patch

import pytest

from stream_pose_ml.services.video_data_merge_service import (
    VideoDataMergeService,
    VideoDataMergeServiceError,
)


class TestVideoDataMergeService:
    """Test the VideoDataMergeService class."""

    @pytest.fixture
    def mock_path_utility(self):
        """Create a mock for path_utility functions."""
        with patch(
            "stream_pose_ml.services.video_data_merge_service.path_utility"
        ) as mock_utility:
            # Set up mock returns for get_file_paths_in_directory
            mock_utility.get_file_paths_in_directory.side_effect = [
                [
                    "/path/to/annotations/video1.json",
                    "/path/to/annotations/video2.json",
                ],  # annotations
                [
                    "/path/to/videos/video1.webm",
                    "/path/to/videos/video2.webm",
                ],  # videos
                [
                    "/path/to/sequences/video1_sequence.json",
                    "/path/to/sequences/video2_sequence.json",
                ],  # sequences
            ]

            # Set up mock returns for get_file_name
            mock_utility.get_file_name.side_effect = (
                lambda path, omit_extension=False: (
                    "video1"
                    if "video1" in path
                    else (
                        "video2"
                        if omit_extension
                        else "video1.json"
                        if "video1" in path
                        else "video2.json"
                    )
                )
            )

            yield mock_utility

    @pytest.fixture
    def mock_transformer(self):
        """Create a mock for AnnotationTransformerService."""
        with patch(
            "stream_pose_ml.services.video_data_merge_service.AnnotationTransformerService"
        ) as mock_transformer:
            mock_instance = MagicMock()
            mock_transformer.return_value = mock_instance

            # Set up mock return for update_video_data_with_annotations
            mock_instance.update_video_data_with_annotations.return_value = (
                ["all_frame1", "all_frame2"],
                ["labeled_frame1"],
                ["unlabeled_frame1"],
            )

            yield mock_transformer

    @pytest.fixture
    def mock_video_data_service(self):
        """Create a mock for VideoDataService."""
        with patch(
            "stream_pose_ml.services.video_data_merge_service.vds.VideoDataService"
        ) as mock_service:
            mock_instance = MagicMock()
            mock_service.return_value = mock_instance

            # Set up mock return for process_video
            mock_instance.process_video.return_value = {
                "name": "video1.webm",
                "frames": {
                    "1": {"frame_number": 1, "data": "test1"},
                    "2": {"frame_number": 2, "data": "test2"},
                },
            }

            yield mock_service

    @pytest.fixture
    def mock_open_json(self):
        """Create a mock for open and json.load."""
        mock = mock_open()
        with patch("builtins.open", mock), patch("json.load") as mock_json_load:
            # Set up mock returns for json.load based on file path
            mock_json_load.side_effect = lambda f: (
                {
                    "annotations": [
                        {"label": "class1", "start_frame": 1, "end_frame": 5}
                    ]
                }  # annotation data
                if "annotation" in str(f.mock_calls)
                else {  # sequence data
                    "name": (
                        "video1.webm"
                        if "video1" in str(f.mock_calls)
                        else "video2.webm"
                    ),
                    "frames": {
                        "1": {"frame_number": 1, "data": "test1"},
                        "2": {"frame_number": 2, "data": "test2"},
                    },
                }
            )

            yield mock

    def test_init_with_videos(self, mock_path_utility, mock_transformer):
        """Test initializing the service with video processing."""
        # Given
        annotations_dir = "/path/to/annotations"
        video_dir = "/path/to/videos"
        keypoints_path = "/path/to/keypoints"

        # When
        service = VideoDataMergeService(
            annotations_data_directory=annotations_dir,
            output_keypoints_path=keypoints_path,
            video_directory=video_dir,
            process_videos=True,
        )

        # Then
        assert service.annotations_data_directory == annotations_dir
        assert service.video_directory == video_dir
        assert service.output_keypoints_path == keypoints_path
        assert service.process_videos is True
        assert isinstance(service.transformer, MagicMock)  # Our mocked transformer

        # Verify create_video_annotation_map was called
        mock_path_utility.get_file_paths_in_directory.assert_any_call(
            directory=annotations_dir, extension="json"
        )
        mock_path_utility.get_file_paths_in_directory.assert_any_call(
            directory=video_dir, extension=["webm", "mp4"]
        )

    def test_init_with_sequences(self, mock_path_utility, mock_transformer):
        """Test initializing the service with sequence data."""
        # Given
        annotations_dir = "/path/to/annotations"
        sequence_dir = "/path/to/sequences"

        # When
        service = VideoDataMergeService(
            annotations_data_directory=annotations_dir,
            sequence_data_directory=sequence_dir,
            process_videos=False,
        )

        # Then
        assert service.annotations_data_directory == annotations_dir
        assert service.sequence_data_directory == sequence_dir
        assert service.process_videos is False

        # Verify create_video_annotation_map was called
        mock_path_utility.get_file_paths_in_directory.assert_any_call(
            directory=annotations_dir, extension="json"
        )
        mock_path_utility.get_file_paths_in_directory.assert_any_call(
            directory=sequence_dir, extension="json"
        )

    def test_create_video_annotation_map(self, mock_path_utility):
        """Test creating the video annotation map."""
        # Given
        annotations_dir = "/path/to/annotations"
        video_dir = "/path/to/videos"
        sequence_dir = "/path/to/sequences"

        # When
        service = VideoDataMergeService(
            annotations_data_directory=annotations_dir,
            video_directory=video_dir,
            sequence_data_directory=sequence_dir,
            process_videos=False,
        )

        # Then
        # Verify maps were created correctly
        assert len(service.annotation_video_map) == 2
        assert len(service.video_annotation_map) == 2
        assert len(service.annotation_sequence_map) == 2
        assert len(service.sequence_annotation_map) == 2

        # Check specific mappings
        assert (
            service.annotation_video_map["/path/to/annotations/video1.json"]
            == "/path/to/videos/video1.webm"
        )
        assert (
            service.annotation_sequence_map["/path/to/annotations/video1.json"]
            == "/path/to/sequences/video1_sequence.json"
        )

    def test_create_video_annotation_map_error(self):
        """Test error when no video directory is specified but process_videos is "
        "True."""
        # Given/When/Then
        with pytest.raises(
            VideoDataMergeServiceError,
            match="No source video directory specified to generate video data from.",
        ):
            VideoDataMergeService(
                annotations_data_directory="/path/to/annotations",
                process_videos=True,
                video_directory=None,
            )

    def test_generate_annotated_video_data_from_sequences(
        self, mock_path_utility, mock_transformer, mock_open_json
    ):
        """Test generating annotated video data from sequence files."""
        # Given
        annotations_dir = "/path/to/annotations"
        sequence_dir = "/path/to/sequences"

        service = VideoDataMergeService(
            annotations_data_directory=annotations_dir,
            sequence_data_directory=sequence_dir,
            process_videos=False,
        )

        # When
        result = service.generate_annotated_video_data()

        # Then
        assert result == {
            "all_frames": [["all_frame1", "all_frame2"], ["all_frame1", "all_frame2"]],
            "labeled_frames": [["labeled_frame1"], ["labeled_frame1"]],
            "unlabeled_frames": [["unlabeled_frame1"], ["unlabeled_frame1"]],
        }

        # Verify transformer was called correctly
        assert (
            mock_transformer.return_value.update_video_data_with_annotations.call_count
            == 2
        )

    def test_generate_annotated_video_data_from_videos(
        self,
        mock_path_utility,
        mock_transformer,
        mock_video_data_service,
        mock_open_json,
    ):
        """Test generating annotated video data from videos."""
        # Given
        annotations_dir = "/path/to/annotations"
        video_dir = "/path/to/videos"
        keypoints_path = "/path/to/keypoints"

        service = VideoDataMergeService(
            annotations_data_directory=annotations_dir,
            video_directory=video_dir,
            output_keypoints_path=keypoints_path,
            process_videos=True,
        )

        # Mock os.path functions for process_video call
        with patch(
            "stream_pose_ml.services.video_data_merge_service.os.path"
        ) as mock_path:
            mock_path.basename.side_effect = lambda path: path.split("/")[-1]
            mock_path.split.side_effect = lambda path: (
                path.rsplit("/", 1)[0],
                path.split("/")[-1],
            )

            # When
            result = service.generate_annotated_video_data()

        # Then
        # Verify VideoDataService was called correctly
        assert mock_video_data_service.return_value.process_video.call_count == 2

        # Verify results
        assert result == {
            "all_frames": [["all_frame1", "all_frame2"]],
            "labeled_frames": [["labeled_frame1"]],
            "unlabeled_frames": [["unlabeled_frame1"]],
        }

    def test_generate_annotated_video_data_with_limit(
        self, mock_path_utility, mock_transformer, mock_open_json
    ):
        """Test generating annotated video data with a limit."""
        # Given
        annotations_dir = "/path/to/annotations"
        sequence_dir = "/path/to/sequences"
        limit = 1

        service = VideoDataMergeService(
            annotations_data_directory=annotations_dir,
            sequence_data_directory=sequence_dir,
            process_videos=False,
        )

        # When
        service.generate_annotated_video_data(limit=limit)

        # Then
        # Verify only one sequence was processed
        assert len(service.merged_data) == 1

        # Verify transformer was called only once
        assert (
            mock_transformer.return_value.update_video_data_with_annotations.call_count
            == 1
        )

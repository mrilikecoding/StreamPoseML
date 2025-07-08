import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parents[3]  # Adjust if needed
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ruff: noqa: E402
from stream_pose_ml.jobs.build_and_format_dataset_job import BuildAndFormatDatasetJob
from stream_pose_ml.learning.dataset import Dataset


class TestBuildAndFormatDatasetJob:
    """Test the BuildAndFormatDatasetJob class."""

    @pytest.fixture
    def mock_video_data_merge_service(self):
        """Create a mock for VideoDataMergeService."""
        with patch(
            "stream_pose_ml.jobs.build_and_format_dataset_job.VideoDataMergeService"
        ) as mock_service:
            mock_instance = MagicMock()
            mock_service.return_value = mock_instance

            # Set up mock data for generate_annotated_video_data
            mock_instance.generate_annotated_video_data.return_value = {
                "all_frames": ["frame1", "frame2"],
                "labeled_frames": ["frame1"],
                "unlabeled_frames": ["frame2"],
            }

            yield mock_service

    @pytest.fixture
    def mock_segmentation_service(self):
        """Create a mock for SegmentationService."""
        with patch(
            "stream_pose_ml.jobs.build_and_format_dataset_job.SegmentationService"
        ) as mock_service:
            mock_instance = MagicMock()
            mock_service.return_value = mock_instance

            # Segmentation returns the same dataset
            mock_instance.segment_dataset.side_effect = lambda dataset: dataset

            yield mock_service

    @pytest.fixture
    def mock_dataset_serializer(self):
        """Create a mock for DatasetSerializer."""
        with patch(
            "stream_pose_ml.jobs.build_and_format_dataset_job.DatasetSerializer"
        ) as mock_serializer:
            mock_instance = MagicMock()
            mock_serializer.return_value = mock_instance

            # Serializer returns sample data
            mock_instance.serialize.return_value = [
                {"id": 1, "features": {"angle": 45.123}},
                {"id": 2, "features": {"angle": 90.456}},
            ]

            yield mock_serializer

    @pytest.fixture
    def mock_pandas(self):
        """Create a mock for pandas functions."""
        with patch("stream_pose_ml.jobs.build_and_format_dataset_job.pd") as mock_pd:
            mock_df = MagicMock()
            mock_pd.json_normalize.return_value = mock_df
            yield mock_pd

    @pytest.fixture
    def mock_round_nested_dict(self):
        """Create a mock for round_nested_dict function."""
        with patch(
            "stream_pose_ml.jobs.build_and_format_dataset_job.round_nested_dict"
        ) as mock_round:
            mock_round.side_effect = lambda item, precision: {
                **item,
                "features": {
                    k: round(v, precision) if isinstance(v, float) else v
                    for k, v in item["features"].items()
                },
            }
            yield mock_round

    def test_build_dataset_from_data_files(self, mock_video_data_merge_service):
        """Test building a dataset from data files."""
        # Given
        annotations_data_directory = "/path/to/annotations"
        sequence_data_directory = "/path/to/sequences"
        limit = 5

        # When
        result = BuildAndFormatDatasetJob.build_dataset_from_data_files(
            annotations_data_directory=annotations_data_directory,
            sequence_data_directory=sequence_data_directory,
            limit=limit,
        )

        # Then
        assert isinstance(result, Dataset)
        assert result.all_frames == ["frame1", "frame2"]
        assert result.labeled_frames == ["frame1"]
        assert result.unlabeled_frames == ["frame2"]

        # Verify VideoDataMergeService was created correctly
        mock_video_data_merge_service.assert_called_once_with(
            annotations_data_directory=annotations_data_directory,
            sequence_data_directory=sequence_data_directory,
            process_videos=False,
        )

        # Verify generate_annotated_video_data was called correctly
        mock_video_data_merge_service.return_value.generate_annotated_video_data.assert_called_once_with(
            limit=limit
        )

    def test_build_dataset_from_videos(self, mock_video_data_merge_service):
        """Test building a dataset directly from videos."""
        # Given
        annotations_directory = "/path/to/annotations"
        video_directory = "/path/to/videos"

        # When
        result = BuildAndFormatDatasetJob.build_dataset_from_videos(
            annotations_directory=annotations_directory, video_directory=video_directory
        )

        # Then
        assert isinstance(result, Dataset)

        # Verify VideoDataMergeService was created correctly
        mock_video_data_merge_service.assert_called_once_with(
            annotations_data_directory=annotations_directory,
            video_directory=video_directory,
            process_videos=True,
        )

    def test_format_dataset(
        self, mock_segmentation_service, mock_dataset_serializer, mock_round_nested_dict
    ):
        """Test formatting a dataset."""
        # Given
        dataset = Dataset(
            all_frames=["frame1", "frame2"],
            labeled_frames=["frame1"],
            unlabeled_frames=["frame2"],
        )

        # When
        result = BuildAndFormatDatasetJob.format_dataset(
            dataset=dataset,
            pool_frame_data_by_clip=True,
            decimal_precision=4,
            include_unlabeled_data=False,
            include_angles=True,
            include_distances=True,
        )

        # Then
        # Verify correct data was returned and rounded
        assert len(result) == 2
        assert result[0]["features"]["angle"] == 45.1230
        assert result[1]["features"]["angle"] == 90.4560

        # Verify SegmentationService was created with correct parameters
        mock_segmentation_service.assert_called_once_with(
            include_unlabeled_data=False,
            segmentation_strategy="none",
            segmentation_splitter_label=None,
            segmentation_window=None,
            segmentation_window_label=None,
        )

        # Verify DatasetSerializer was created with correct parameters
        mock_dataset_serializer.assert_called_once_with(
            pool_rows=True,
            include_normalized=True,
            include_angles=True,
            include_distances=True,
            include_joints=False,
            include_z_axis=False,
        )

    def test_format_dataset_without_rounding(
        self, mock_segmentation_service, mock_dataset_serializer
    ):
        """Test formatting a dataset without decimal rounding."""
        # Given
        dataset = Dataset(
            all_frames=["frame1", "frame2"],
            labeled_frames=["frame1"],
            unlabeled_frames=["frame2"],
        )

        # When
        result = BuildAndFormatDatasetJob.format_dataset(
            dataset=dataset, decimal_precision=None
        )

        # Then
        # Verify original values were returned without rounding
        assert len(result) == 2
        assert result[0]["features"]["angle"] == 45.123
        assert result[1]["features"]["angle"] == 90.456

    def test_write_dataset_to_csv(self, mock_pandas):
        """Test writing a dataset to CSV."""
        # Given
        csv_location = "/path/to/output"
        formatted_dataset = [
            {"id": 1, "features": {"angle": 45.123}},
            {"id": 2, "features": {"angle": 90.456}},
        ]
        filename = "test_dataset"

        # When
        result = BuildAndFormatDatasetJob.write_dataset_to_csv(
            csv_location=csv_location,
            formatted_dataset=formatted_dataset,
            filename=filename,
        )

        # Then
        assert result is True

        # Verify pandas.json_normalize was called correctly
        mock_pandas.json_normalize.assert_called_once_with(data=formatted_dataset)

        # Verify to_csv was called with correct path
        mock_pandas.json_normalize.return_value.to_csv.assert_called_once_with(
            f"{csv_location}/{filename}.csv"
        )

    @patch("stream_pose_ml.jobs.build_and_format_dataset_job.time")
    def test_write_dataset_to_csv_with_timestamp(self, mock_time, mock_pandas):
        """Test writing a dataset to CSV with timestamp filename."""
        # Given
        csv_location = "/path/to/output"
        formatted_dataset = [{"id": 1}]
        mock_time.time_ns.return_value = 123456789

        # When
        BuildAndFormatDatasetJob.write_dataset_to_csv(
            csv_location=csv_location, formatted_dataset=formatted_dataset
        )

        # Then
        # Verify to_csv was called with timestamp filename
        mock_pandas.json_normalize.return_value.to_csv.assert_called_once_with(
            f"{csv_location}/dataset_123456789.csv"
        )

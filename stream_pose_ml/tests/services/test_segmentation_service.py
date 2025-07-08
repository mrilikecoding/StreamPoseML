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

from stream_pose_ml.services.segmentation_service import (
    SegmentationService,
    SegmentationServiceError,
    SegmentationStrategy,
)


class TestSegmentationService:
    """Test the SegmentationService class."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock for Dataset."""
        mock = MagicMock()
        # Set up mock frame data
        mock.all_frames = [
            [  # Video 1
                {"category": "class1", "data": {"frame_number": 1, "value": "data1"}},
                {"category": "class1", "data": {"frame_number": 2, "value": "data2"}},
                {"category": "class2", "data": {"frame_number": 3, "value": "data3"}},
                {"category": "class2", "data": {"frame_number": 4, "value": "data4"}},
                {"category": "class1", "data": {"frame_number": 5, "value": "data5"}},
            ],
            [  # Video 2
                {"category": "class1", "data": {"frame_number": 1, "value": "data6"}},
                {"category": "class2", "data": {"frame_number": 2, "value": "data7"}},
                {"category": "class2", "data": {"frame_number": 3, "value": "data8"}},
            ],
        ]

        mock.labeled_frames = mock.all_frames  # All frames are labeled in this test
        mock.unlabeled_frames = []  # No unlabeled frames for simplicity

        yield mock

    @pytest.fixture
    def mock_labeled_clip(self):
        """Create a mock for LabeledClip."""
        with patch(
            "stream_pose_ml.services.segmentation_service.LabeledClip"
        ) as mock_clip:
            mock_clip.side_effect = lambda frames: MagicMock(frames=frames)
            yield mock_clip

    def test_init(self):
        """Test initializing the service."""
        # Given/When
        service = SegmentationService(
            segmentation_strategy="window",
            include_unlabeled_data=True,
            segmentation_window=5,
            segmentation_splitter_label="category",
            segmentation_window_label="action",
        )

        # Then
        assert service.segemetation_strategy == SegmentationStrategy.WINDOW
        assert service.segmentation_window == 5
        assert service.segmentation_splitter_label == "category"
        assert service.segmentation_window_label == "action"
        assert service.include_unlabeled_data is True
        assert service.merged_data == []

    def test_segment_all_frames(self, mock_dataset, mock_labeled_clip):
        """Test segmenting all frames individually."""
        # Given
        service = SegmentationService(
            segmentation_strategy="none", include_unlabeled_data=True
        )

        # When
        result = service.segment_all_frames(mock_dataset)

        # Then
        # There should be 8 clips - one for each frame in both videos
        assert len(result) == 8

        # Verify LabeledClip was called for each frame
        assert mock_labeled_clip.call_count == 8

        # Verify the first clip has the first frame
        assert result[0].frames == [mock_dataset.all_frames[0][0]]

    def test_segment_all_frames_labeled_only(self, mock_dataset, mock_labeled_clip):
        """Test segmenting only labeled frames."""
        # Given
        service = SegmentationService(
            segmentation_strategy="none", include_unlabeled_data=False
        )

        # Prepare mock dataset with some unlabeled frames
        mock_dataset.labeled_frames = [
            mock_dataset.all_frames[0][:3],  # First 3 frames of video 1
            mock_dataset.all_frames[1][:2],  # First 2 frames of video 2
        ]

        # When
        result = service.segment_all_frames(mock_dataset)

        # Then
        # There should be 5 clips - one for each labeled frame
        assert len(result) == 5

        # Verify LabeledClip was called for each labeled frame
        assert mock_labeled_clip.call_count == 5

    def test_split_on_label(self, mock_dataset, mock_labeled_clip):
        """Test splitting frames based on label changes."""
        # Given
        service = SegmentationService(
            segmentation_strategy="split_on_label",
            segmentation_splitter_label="category",
        )

        # When
        result = service.split_on_label(mock_dataset)

        # Then
        # There should be clips for each group of frames with the same label
        assert len(result) > 0

        # Let's check the structure of the first clip
        assert hasattr(result[0], "frames"), "LabeledClip should have frames attribute"
        assert isinstance(result[0].frames, list), "frames should be a list"

        # Check that each clip has at least one frame
        for clip in result:
            assert len(clip.frames) > 0, "Each clip should have at least one frame"

    def test_split_on_label_with_window(self, mock_dataset, mock_labeled_clip):
        """Test splitting frames with a window size limit."""
        # Given
        service = SegmentationService(
            segmentation_strategy="split_on_label",
            segmentation_splitter_label="category",
            segmentation_window=1,  # Only take the last frame of each segment
        )

        # When
        result = service.split_on_label(mock_dataset)

        # Then
        # There should be clips for each group of frames with the same label
        assert len(result) > 0

        # Each clip should have only 1 frame when using a window size of 1
        for clip in result:
            assert len(clip.frames) == 1, (
                "Each clip should have exactly 1 frame with window size of 1"
            )

    def test_split_on_label_error(self, mock_dataset):
        """Test error when no segmentation_splitter_label is provided."""
        # Given
        service = SegmentationService(
            segmentation_strategy="split_on_label", segmentation_splitter_label=None
        )

        # When/Then
        with pytest.raises(
            SegmentationServiceError,
            match="segmentation_spliiter_label must be present",
        ):
            service.split_on_label(mock_dataset)

    def test_split_on_window(self, mock_dataset, mock_labeled_clip):
        """Test splitting frames based on a fixed window size."""
        # Given
        service = SegmentationService(
            segmentation_strategy="window",
            segmentation_window=2,
            segmentation_window_label="category",
        )

        # When
        result = service.split_on_window(mock_dataset)

        # Then
        # There should be clips for every frame that's the end of a 2-frame window
        # and has a label
        assert len(result) > 0

        # Each clip should have 2 frames (the window size)
        for clip in result:
            assert len(clip.frames) == 2

    def test_split_on_window_error(self, mock_dataset):
        """Test error when required parameters are missing."""
        # Given
        service = SegmentationService(
            segmentation_strategy="window",
            segmentation_window=None,
            segmentation_window_label=None,
        )

        # When/Then
        with pytest.raises(
            SegmentationServiceError,
            match="Both segmentation window and segmentation window label is required",
        ):
            service.split_on_window(mock_dataset)

    def test_flatten_into_columns(self, mock_dataset, mock_labeled_clip):
        """Test flattening frames into columns."""
        # Given
        service = SegmentationService(
            segmentation_strategy="flatten_into_columns",
            segmentation_window=2,
            segmentation_window_label="category",
        )

        # When
        result = service.flatten_into_columns(mock_dataset)

        # Then
        # There should be clips for every frame that's the end of a 2-frame window
        # and has a label
        assert len(result) > 0

        # Each clip should have just 1 frame (which is a flattened representation)
        for clip in result:
            assert len(clip.frames) == 1
            # The frames should have a data key that contains flattened values
            assert "data" in clip.frames[0]

    def test_flatten_into_columns_error(self, mock_dataset):
        """Test error when required parameters are missing."""
        # Given
        service = SegmentationService(
            segmentation_strategy="flatten_into_columns",
            segmentation_window=None,
            segmentation_window_label=None,
        )

        # When/Then
        with pytest.raises(
            SegmentationServiceError,
            match="Both segmentation window and segmentation window label is required",
        ):
            service.flatten_into_columns(mock_dataset)

    def test_flatten_segment_into_row(self):
        """Test flattening a segment of frames into a single row."""
        # Given
        service = SegmentationService(
            segmentation_strategy="flatten_into_columns",
            segmentation_window=2,
            segmentation_window_label="category",
        )

        frame_segment = [
            {
                "category": "class1",
                "data": {
                    "frame_number": 1,
                    "joints": {"joint1": {"x": 1, "y": 2}},
                    "angles": {"angle1": 45},
                },
            },
            {
                "category": "class1",
                "data": {
                    "frame_number": 2,
                    "joints": {"joint1": {"x": 3, "y": 4}},
                    "angles": {"angle1": 90},
                },
            },
        ]

        # When
        result = service.flatten_segment_into_row(frame_segment)

        # Then
        # Result should have the category from the last frame
        assert result["category"] == "class1"

        # Data should be flattened with frame-specific keys
        assert "data" in result
        assert "joints" in result["data"]
        assert "angles" in result["data"]

        # Check specific flattened values
        assert result["data"]["joints"]["frame-1-joint1"] == {"x": 1, "y": 2}
        assert result["data"]["joints"]["frame-2-joint1"] == {"x": 3, "y": 4}
        assert result["data"]["angles"]["frame-1-angle1"] == 45
        assert result["data"]["angles"]["frame-2-angle1"] == 90

    def test_flatten_on_example(self, mock_dataset, mock_labeled_clip):
        """Test flattening frames based on a label change and then flattening into
        columns."""
        # Given
        service = SegmentationService(
            segmentation_strategy="flatten_on_example",
            segmentation_splitter_label="category",
            segmentation_window=1,  # Just take last frame of each segment
        )

        # When
        with patch.object(service, "split_on_label") as mock_split:
            service.flatten_on_example(mock_dataset)

            # Then
            # Should call split_on_label with flatten_into_columns=True
            mock_split.assert_called_once_with(
                dataset=mock_dataset, flatten_into_columns=True
            )

    def test_segment_dataset_none(self, mock_dataset):
        """Test segmenting a dataset with 'none' strategy."""
        # Given
        service = SegmentationService(
            segmentation_strategy="none", include_unlabeled_data=True
        )

        # When
        with patch.object(
            service, "segment_all_frames", return_value=["clip1", "clip2"]
        ) as mock_segment:
            result = service.segment_dataset(mock_dataset)

            # Then
            mock_segment.assert_called_once_with(dataset=mock_dataset)
            assert result.segmented_data == ["clip1", "clip2"]

    def test_segment_dataset_split_on_label(self, mock_dataset):
        """Test segmenting a dataset with 'split_on_label' strategy."""
        # Given
        service = SegmentationService(
            segmentation_strategy="split_on_label",
            segmentation_splitter_label="category",
        )

        # When
        with patch.object(
            service, "split_on_label", return_value=["clip1", "clip2"]
        ) as mock_segment:
            result = service.segment_dataset(mock_dataset)

            # Then
            mock_segment.assert_called_once_with(dataset=mock_dataset)
            assert result.segmented_data == ["clip1", "clip2"]

    def test_segment_dataset_window(self, mock_dataset):
        """Test segmenting a dataset with 'window' strategy."""
        # Given
        service = SegmentationService(
            segmentation_strategy="window",
            segmentation_window=2,
            segmentation_window_label="category",
        )

        # When
        with patch.object(
            service, "split_on_window", return_value=["clip1", "clip2"]
        ) as mock_segment:
            result = service.segment_dataset(mock_dataset)

            # Then
            mock_segment.assert_called_once_with(dataset=mock_dataset)
            assert result.segmented_data == ["clip1", "clip2"]

    def test_segment_dataset_flatten_into_columns(self, mock_dataset):
        """Test segmenting a dataset with 'flatten_into_columns' strategy."""
        # Given
        service = SegmentationService(
            segmentation_strategy="flatten_into_columns",
            segmentation_window=2,
            segmentation_window_label="category",
        )

        # When
        with patch.object(
            service, "flatten_into_columns", return_value=["clip1", "clip2"]
        ) as mock_segment:
            result = service.segment_dataset(mock_dataset)

            # Then
            mock_segment.assert_called_once_with(dataset=mock_dataset)
            assert result.segmented_data == ["clip1", "clip2"]

    def test_segment_dataset_flatten_on_example(self, mock_dataset):
        """Test segmenting a dataset with 'flatten_on_example' strategy."""
        # Given
        service = SegmentationService(
            segmentation_strategy="flatten_on_example",
            segmentation_splitter_label="category",
        )

        # When
        with patch.object(
            service, "flatten_on_example", return_value=["clip1", "clip2"]
        ) as mock_segment:
            result = service.segment_dataset(mock_dataset)

            # Then
            mock_segment.assert_called_once_with(dataset=mock_dataset)
            assert result.segmented_data == ["clip1", "clip2"]

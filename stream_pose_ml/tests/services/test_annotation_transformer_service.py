import os
import sys
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parents[
    3
]  # /Users/nathangreen/Development/stream_pose_ml
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ruff: noqa: E402
from stream_pose_ml.services.annotation_transformer_service import (
    AnnotationTransformerService,
    find_project_root,
    get_nested_key,
)


class TestGetNestedKey:
    """Test the get_nested_key function."""

    def test_get_nested_key_single_level(self):
        """Test getting a key at a single level."""
        data = {"level1": "value1"}
        result = get_nested_key(data, "level1")
        assert result == "value1"

    def test_get_nested_key_multiple_levels(self):
        """Test getting a key at multiple levels."""
        data = {"level1": {"level2": {"level3": "value3"}}}
        result = get_nested_key(data, "level1.level2.level3")
        assert result == "value3"

    def test_get_nested_key_missing_key(self):
        """Test getting a missing key raises an exception."""
        data = {"level1": {"level2": "value2"}}
        with pytest.raises(KeyError):
            get_nested_key(data, "level1.missing")


class TestFindProjectRoot:
    """Test the find_project_root function."""

    @patch("os.path.abspath")
    @patch("os.listdir")
    @patch("os.path.dirname")
    def test_find_project_root_success(self, mock_dirname, mock_listdir, mock_abspath):
        """Test finding the project root successfully."""
        # Set up mocks
        mock_abspath.return_value = "/path/to/current"
        mock_listdir.return_value = ["file1", "config.yml", "file2"]

        # When
        result = find_project_root()

        # Then
        assert result == "/path/to/current"
        mock_abspath.assert_called_once_with(os.curdir)
        mock_listdir.assert_called_once_with("/path/to/current")
        # dirname should not be called since we found the file
        mock_dirname.assert_not_called()

    @patch("os.path.abspath")
    @patch("os.listdir")
    @patch("os.path.dirname")
    def test_find_project_root_traverse_up(
        self, mock_dirname, mock_listdir, mock_abspath
    ):
        """Test finding the project root by traversing up directories."""
        # Set up mocks
        mock_abspath.return_value = "/path/to/current"
        mock_listdir.side_effect = [
            ["file1", "file2"],  # First directory
            ["file3", "config.yml", "file4"],  # Parent directory
        ]
        mock_dirname.return_value = "/path/to"

        # When
        result = find_project_root()

        # Then
        assert result == "/path/to"
        mock_abspath.assert_called_once_with(os.curdir)
        assert mock_listdir.call_count == 2
        mock_dirname.assert_called_once_with("/path/to/current")

    @patch("os.path.abspath")
    @patch("os.listdir")
    @patch("os.path.dirname")
    def test_find_project_root_not_found(
        self, mock_dirname, mock_listdir, mock_abspath
    ):
        """Test exception when project root is not found."""
        # Set up mocks
        mock_abspath.return_value = "/path"
        mock_listdir.return_value = ["file1", "file2"]
        mock_dirname.side_effect = (
            lambda path: path
        )  # Return the same path to simulate root

        # When/Then
        with pytest.raises(
            Exception, match="Root directory with config.yml not found!"
        ):
            find_project_root()


class TestAnnotationTransformerService:
    """Test the AnnotationTransformerService class."""

    @pytest.fixture
    def mock_find_project_root(self):
        """Create a mock for find_project_root."""
        with patch(
            "stream_pose_ml.services.annotation_transformer_service.find_project_root"
        ) as mock_find:
            mock_find.return_value = "/path/to/project"
            yield mock_find

    @pytest.fixture
    def mock_yaml(self):
        """Create a mock for yaml."""
        with patch(
            "stream_pose_ml.services.annotation_transformer_service.yaml"
        ) as mock_yaml:
            mock_yaml.load.return_value = {
                "annotation_schema": {
                    "annotations_key": "annotations",
                    "annotation_fields": {
                        "label": "label",
                        "start_frame": "start_frame",
                        "end_frame": "end_frame",
                    },
                    "label_class_mapping": {
                        "class1": "category1",
                        "class2": "category2",
                    },
                }
            }
            yield mock_yaml

    @pytest.fixture
    def mock_open_file(self):
        """Create a mock for open."""
        mock = mock_open(read_data="mock_file_content")
        with patch("builtins.open", mock):
            yield mock

    def test_load_annotation_schema(
        self, mock_find_project_root, mock_yaml, mock_open_file
    ):
        """Test loading the annotation schema."""
        # Given
        expected_schema = {
            "annotations_key": "annotations",
            "annotation_fields": {
                "label": "label",
                "start_frame": "start_frame",
                "end_frame": "end_frame",
            },
            "label_class_mapping": {"class1": "category1", "class2": "category2"},
        }

        # When
        result = AnnotationTransformerService.load_annotation_schema()

        # Then
        assert result == expected_schema
        mock_find_project_root.assert_called_once()
        mock_open_file.assert_called_once_with("/path/to/project/config.yml")
        mock_yaml.load.assert_called_once_with(
            mock_open_file.return_value, Loader=mock_yaml.FullLoader
        )

    def test_update_video_data_with_annotations(self):
        """Test updating video data with annotations."""
        # Given
        annotation_data = {
            "annotations": [
                {"label": "class1", "start_frame": 1, "end_frame": 5},
                {"label": "class2", "start_frame": 3, "end_frame": 7},
            ]
        }

        video_data = {
            "name": "test_video",
            "frames": {
                "1": {"frame_number": 1, "data": "frame1_data"},
                "2": {"frame_number": 2, "data": "frame2_data"},
                "3": {"frame_number": 3, "data": "frame3_data"},
                "4": {"frame_number": 4, "data": "frame4_data"},
                "5": {"frame_number": 5, "data": "frame5_data"},
                "6": {"frame_number": 6, "data": "frame6_data"},
            },
        }

        schema = {
            "annotations_key": "annotations",
            "annotation_fields": {
                "label": "label",
                "start_frame": "start_frame",
                "end_frame": "end_frame",
            },
            "label_class_mapping": {"class1": "category1", "class2": "category2"},
        }

        # When
        all_frames, labeled_frames, unlabeled_frames = (
            AnnotationTransformerService.update_video_data_with_annotations(
                annotation_data=annotation_data, video_data=video_data, schema=schema
            )
        )

        # Then
        # Verify all frames were processed
        assert len(all_frames) == 6

        # Frames 3-5 should be labeled with both classes
        assert len(labeled_frames) == 3  # Frames 3, 4, 5 have both labels

        # Frames 1-2 and 6 should be unlabeled (missing at least one class)
        assert len(unlabeled_frames) == 3

        # Check the structure of a labeled frame
        labeled_frame = labeled_frames[0]
        assert labeled_frame["category1"] == "class1"
        assert labeled_frame["category2"] == "class2"
        assert labeled_frame["data"]["frame_number"] == 3
        assert labeled_frame["video_id"] == "test_video"

    @patch(
        "stream_pose_ml.services.annotation_transformer_service.AnnotationTransformerService.load_annotation_schema"
    )
    def test_update_video_data_with_annotations_no_schema(self, mock_load_schema):
        """Test updating video data with annotations when no schema is provided."""
        # Given
        mock_load_schema.return_value = {
            "annotations_key": "annotations",
            "annotation_fields": {
                "label": "label",
                "start_frame": "start_frame",
                "end_frame": "end_frame",
            },
            "label_class_mapping": {"class1": "category1"},
        }

        annotation_data = {
            "annotations": [{"label": "class1", "start_frame": 1, "end_frame": 5}]
        }

        video_data = {
            "name": "test_video",
            "frames": {"1": {"frame_number": 1, "data": "frame1_data"}},
        }

        # When
        all_frames, labeled_frames, unlabeled_frames = (
            AnnotationTransformerService.update_video_data_with_annotations(
                annotation_data=annotation_data, video_data=video_data
            )
        )

        # Then
        mock_load_schema.assert_called_once()
        assert len(all_frames) == 1
        assert len(labeled_frames) == 1
        assert len(unlabeled_frames) == 0

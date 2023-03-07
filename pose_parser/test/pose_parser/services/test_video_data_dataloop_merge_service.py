import unittest
from pathlib import Path

from pose_parser.services.video_data_dataloop_merge_service import (
    VideoDataDataloopMergeService,
    VideoDataDataloopMergeServiceError,
)


class TestVideoDataDataloopMergeService(unittest.TestCase):
    def setUp(self) -> None:
        self.annotations_directory = "./source_annotations"
        self.source_videos = "./source_videos"
        self.output_data_path = "./data/generated_datasets"

        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_create_video_annotation_map(self):
        """
        GIVEN annotation and video paths
        WHEN calling to create a map
        THEN a map dict is created where each entry reference the correct matching annotation / video
        """
        vdms = VideoDataDataloopMergeService(
            annotations_directory=self.annotations_directory,
            video_directory=self.source_videos,
            output_data_path=self.output_data_path,
        )

        success = vdms.create_video_annotation_map()
        self.assertTrue(success)
        for k, v in vdms.annotation_video_map.items():
            self.assertIn(Path(k).stem, v)

    def test_generate_dataset_from_map(self):
        """
        GIVEN a video dataloop merge service instance with a file map
        WHEN calling to generate a dataset from the file map
        THEN a dictionary object is returned keyed off all labels found in annotation data with culled example clip data for each label
        """
        vdms = VideoDataDataloopMergeService(
            annotations_directory=self.annotations_directory,
            video_directory=self.source_videos,
            output_data_path=self.output_data_path,
        )
        vdms.create_video_annotation_map()
        dataset = vdms.generate_dataset_from_map(limit=2)
        self.assertIsInstance(dataset, dict)
        for key in ["Left Step", "Right Step", "Successful Weight Transfer"]:
            self.assertIn(key, dataset)
            self.assertIsInstance(dataset[key], list)

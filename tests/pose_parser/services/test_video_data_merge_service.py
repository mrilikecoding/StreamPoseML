import unittest
import os
import shutil
from pathlib import Path

from stream_pose_ml.services.video_data_merge_service import (
    VideoDataMergeService,
    VideoDataMergeServiceError,
)


class TestVideoDataMergeService(unittest.TestCase):
    def setUp(self) -> None:
        self.annotations_directory = "./data/source_annotations"
        self.source_videos = "./data/source_videos"
        self.output_data_path = "./data/generated_datasets"
        self.output_keypoints_path = "./data/keypoints"

        return super().setUp()

    def tearDown(self) -> None:
        # cleanup
        try:
            shutil.rmtree(self.output_data_path)
            shutil.rmtree(self.output_keypoints_path)
        except:
            return super().tearDown()
        return super().tearDown()

    def test_create_video_annotation_map(self):
        """
        GIVEN annotation and video paths
        WHEN calling to create a map
        THEN a map dict is created where each entry reference the correct matching annotation / video
        """
        vdms = VideoDataMergeService(
            annotations_directory=self.annotations_directory,
            video_directory=self.source_videos,
            output_data_path=self.output_data_path,
            output_keypoints_path=self.output_keypoints_path,
        )

        success = vdms.create_video_annotation_map()
        self.assertTrue(success)
        for k, v in vdms.annotation_video_map.items():
            self.assertIn(Path(k).stem, v)

    def test_generate_dataset(self):
        """
        GIVEN a video annotation merge service instance with a file map
        WHEN calling to generate a dataset from the file map
        THEN a dictionary object is returned keyed off all labels found in annotation data with culled example clip data for each label
        """
        vdms = VideoDataMergeService(
            annotations_directory=self.annotations_directory,
            video_directory=self.source_videos,
            output_data_path=self.output_data_path,
            output_keypoints_path=self.output_keypoints_path,
        )
        dataset = vdms.generate_dataset(limit=2)
        self.assertIsInstance(dataset, dict)
        for key in ["Left Step", "Right Step", "Successful Weight Transfer"]:
            self.assertIn(key, dataset)
            self.assertIsInstance(dataset[key], list)

    def test_write_merged_data_to_file(self):
        """
        GIVEN a merge service instance that has created a annotation map
        WHEN write to file is selected when generating a dataset
        THEN a file is successfully created in the specified location
        """
        limit = 1
        vdms = VideoDataMergeService(
            annotations_directory=self.annotations_directory,
            video_directory=self.source_videos,
            output_data_path=self.output_data_path,
            output_keypoints_path=self.output_keypoints_path,
        )
        vdms.generate_dataset(limit=limit, write_to_file=True)
        self.assertEqual(True, os.path.exists(self.output_data_path))

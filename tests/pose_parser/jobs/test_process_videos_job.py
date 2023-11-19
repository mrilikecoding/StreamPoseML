import unittest
import shutil
import time


import yaml

CONFIG = yaml.safe_load(open("./config.yml"))

from stream_pose_ml.jobs.process_videos_job import ProcessVideosJob


class TestProcessVideosJob(unittest.TestCase):
    """
    This class is for testing the ProcessVideosJob class

    The idea will be to run this as a job queue, but this is
    step one.
    """

    def setUp(self) -> None:
        self.output_keypoints_data_path = CONFIG["test_keypoints_data_output_directory"]
        self.output_sequence_data_path = CONFIG["test_sequence_data_output_directory"]
        self.video_path = CONFIG["test_video_directory"]
        return super().setUp()

    def tearDown(self) -> None:
        # cleanup
        try:
            shutil.rmtree(self.output_keypoints_data_path)
            shutil.rmtree(self.output_sequence_data_path)
        except:
            return super().tearDown()

        return super().tearDown()

    def test_process_videos(self):
        """
        GIVEN a ProcessVideoJob class
        WHEN passed a valid source video directory
        THEN videos are processed up to a passed limit
        """
        limit = 2
        src_videos_path = self.video_path
        output_keypoints_data_path = self.output_keypoints_data_path
        output_sequence_data_path = self.output_sequence_data_path

        # TODO prob want a nice way to do this outside this test :)
        # To run the real vids uncomment this to override
        # limit = None
        src_videos_path = CONFIG["source_video_directory"]
        folder = f"run-{time.time_ns()}"  # give a timestamped folder to not overwrite
        output_keypoints_data_path = (
            f'{CONFIG["keypoints_data_output_directory"]}/{folder}'
        )
        output_sequence_data_path = (
            f'{CONFIG["sequence_data_output_directory"]}/{folder}'
        )

        results = ProcessVideosJob().process_videos(
            src_videos_path=src_videos_path,
            output_keypoints_data_path=output_keypoints_data_path,
            output_sequence_data_path=output_sequence_data_path,
            write_keypoints_to_file=True,
            write_serialized_sequence_to_file=True,
            limit=limit,
            configuration={},
            preprocess_video=True,
        )
        if limit:
            self.assertEqual(len(results), limit)

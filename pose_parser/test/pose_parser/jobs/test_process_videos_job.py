import unittest
import shutil

from pose_parser.jobs.process_videos_job import ProcessVideosJob


class TestProcessVideosJob(unittest.TestCase):
    """
    This class is for testing the ProcessVideosJob class

    The idea will be to run this as a job queue, but this is
    step one.
    """

    def setUp(self) -> None:
        self.output_path = "./data/keypoints"
        self.video_path = "./source_videos"
        self.cleanup = False
        return super().setUp()

    def tearDown(self) -> None:
        # cleanup
        if self.cleanup:
            try:
                shutil.rmtree(self.output_path)
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
        output_data_path = self.output_path
        results = ProcessVideosJob().process_videos(
            src_videos_path=src_videos_path,
            output_data_path=output_data_path,
            limit=limit,
            write_to_file=True
        )
        self.assertEqual(len(results), limit)

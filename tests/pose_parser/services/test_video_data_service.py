import unittest
import shutil

class TestVideoDataService(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.output_data_path = "./stream_pose_ml/tmp/data/keypoints"
        self.video_input_path = "./stream_pose_ml/test_videos"
        self.input_filename = "front.mp4"

    @classmethod
    def tearDownClass(self) -> None:
        # cleanup
        try:
            shutil.rmtree(self.output_data_path)
        except:
            return super().tearDown(self)

        return super().tearDown(self)

    def test_process_video(self):
        vds = VideoDataService()
        data = vds.process_video(
            input_filename=self.input_filename,
            video_input_path=self.video_input_path,
            output_data_path=self.output_data_path,
            include_geometry=True,
        )
        # TODO assert we have expected keys
        # TODO update frame serializer with right data

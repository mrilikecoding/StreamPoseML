import unittest


from pose_parser.video_data_service import VideoDataService


class TestVideoDataService(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.output_data_path = "./tmp/data/keypoints"
        self.video_input_path = "./test_videos"
        self.input_filename = "front.mp4"

    def test_process_video(self):
        vds = VideoDataService()
        data = vds.process_video(
            input_filename=self.input_filename,
            video_input_path=self.video_input_path,
            output_data_path=self.output_data_path,
        )
        # TODO assert we have expected keys
        # TODO update frame serializer with right data

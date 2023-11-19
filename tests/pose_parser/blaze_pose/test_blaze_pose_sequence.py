import unittest
import shutil

from stream_pose_ml.blaze_pose.mediapipe_client import MediaPipeClient
from stream_pose_ml.blaze_pose.blaze_pose_sequence import (
    BlazePoseSequence,
    BlazePoseSequenceError,
)
from stream_pose_ml.blaze_pose.blaze_pose_frame import BlazePoseFrame


class TestBlazePoseSequence(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.output_path = "./stream_pose_ml/tmp/data/keypoints"
        self.video_path = "./stream_pose_ml/test_videos"
        input_path = self.video_path
        output_path = self.output_path
        mpc = MediaPipeClient(
            video_input_filename="back.mp4",
            video_input_path=input_path,
            video_output_prefix=output_path,
        )
        mpc.process_video(limit=50)
        self.mpc = mpc
        self.frame_data_list = mpc.frame_data_list

    @classmethod
    def tearDownClass(self) -> None:
        # cleanup
        try:
            shutil.rmtree(self.output_path)
        except:
            return super().tearDown(self)

        return super().tearDown(self)

    def test_init(self):
        """
        GIVEN a BlazePoseSequence class
        WHEN initialized with correct frame data list
        THEN an object is successfully initialized with a sequence property
        """
        bps = BlazePoseSequence(name="test", sequence=self.frame_data_list)
        self.assertEqual(bps.sequence_data, self.frame_data_list)

    def test_client_init_error(self):
        """
        GIVEN a BlasePoseSequence class
        WHEN initialized with bad data
        THEN a BlazePoseSequenceError is raised
        """
        bad_data = [{"frame_number": 0}]
        self.assertRaises(
            BlazePoseSequenceError,
            lambda: BlazePoseSequence(name="test", sequence=bad_data),
        )

    def test_validate_pose_schema(self):
        """
        GIVEN a BlazePoseSequence object
        WHEN validate_pose_schema is called
        THEN True is returned if the required keys are present
        """
        bps = BlazePoseSequence(name="test", sequence=self.frame_data_list)
        frame_data_no_joint = self.frame_data_list[0]
        frame_data_joint = self.frame_data_list[2]
        result = bps.validate_pose_schema(frame_data=frame_data_no_joint)
        self.assertEqual(True, result)
        result = bps.validate_pose_schema(frame_data=frame_data_joint)
        self.assertEqual(True, result)

    def test_validate_pose_schema_invalid_data(self):
        """
        GIVEN a BlazePoseSequence object
        WHEN validate_pose_schema is called
        THEN True is returned if the required keys are present
        """
        bps = BlazePoseSequence(name="test", sequence=self.frame_data_list)
        bad_data = {"frame_number": 0}
        self.assertRaises(
            BlazePoseSequenceError,
            lambda: bps.validate_pose_schema(frame_data=bad_data),
        )

    def test_generate_pose_frames_from_sequence(self):
        bps = BlazePoseSequence(name="test", sequence=self.frame_data_list)
        bps.generate_blaze_pose_frames_from_sequence()
        self.assertEqual(len(bps.frames), len(self.frame_data_list))
        for frame in bps.frames:
            self.assertIsInstance(frame, BlazePoseFrame)

import unittest
import shutil
import time
from pathlib import Path
import os
import json

from stream_pose_ml.blaze_pose.mediapipe_client import (
    MediaPipeClient,
    MediaPipeClientError,
)


class TestMediaPipeClient(unittest.TestCase):
    def setUp(self) -> None:
        self.output_path = "./stream_pose_ml/tmp/data/keypoints"
        self.video_path = "./stream_pose_ml/test_videos"
        return super().setUp()

    def tearDown(self) -> None:
        # cleanup
        try:
            shutil.rmtree(self.output_path)
        except:
            return super().tearDown()

        return super().tearDown()

    def test_client_init(self):
        """
        GIVEN input/output path and filenames
        WHEN passed into the MediaPipeClient at init
        THEN the client object sets the appropriate output data paths with the current timestamp
        """
        input_path = self.video_path
        output_path = self.output_path
        files = ["back.mp4", "front.mp4", "side.mp4"]
        for f in files:
            id = int(time.time_ns())
            mpc = MediaPipeClient(
                video_input_filename=f,
                video_input_path=input_path,
                video_output_prefix=output_path,
                id=id,
            )
            self.assertEqual(mpc.json_output_path, f"{output_path}/{Path(f).stem}-{id}")

    def test_process_video(self):
        """
        GIVEN a file passed into a MediaPipeClient
        WHEN process video is called
        THEN pose landmarks are added to the client object
        """
        input_path = self.video_path
        output_path = self.output_path
        mpc = MediaPipeClient(
            video_input_filename="back.mp4",
            video_input_path=input_path,
            video_output_prefix=output_path,
        )
        mpc.process_video(limit=50)
        self.assertEqual(mpc.frame_count, len(mpc.frame_data_list))
        for pose_data in mpc.frame_data_list:
            self.assertIn("sequence_id", pose_data)
            self.assertIn("sequence_source", pose_data)
            self.assertIn("frame_number", pose_data)
            self.assertIn("joint_positions", pose_data)
            self.assertIn("image_dimensions", pose_data)

    def test_serialize_pose_landmarks(self):
        """
        GIVEN a MediaPipe client and some extracted pose landmarks from a frame
        WHEN the pose landmarks are serialized
        THEN a dictionary is returned containing the appropriate data
        """
        limit = 10
        input_path = self.video_path
        output_path = self.output_path
        mpc = MediaPipeClient(
            video_input_filename="back.mp4",
            video_input_path=input_path,
            video_output_prefix=output_path,
        )
        mpc.process_video(limit=limit)
        raw_results = mpc._results_raw
        for pose_object in raw_results:
            if pose_object.pose_landmarks:
                landmarks = list(pose_object.pose_landmarks.landmark)
                serialized = mpc.serialize_pose_landmarks(landmarks)
                for joint in mpc.joints:
                    self.assertIn("x", serialized[joint])
                    self.assertIn("y", serialized[joint])
                    self.assertIn("z", serialized[joint])
                    self.assertIn("x_normalized", serialized[joint])
                    self.assertIn("y_normalized", serialized[joint])
                    self.assertIn("z_normalized", serialized[joint])

    def test_write_pose_data_to_file(self):
        """
        GIVEN a MediaPipeClient with a processed video
        WHEN pose data is written to files
        THEN an amount of files equal to the passed limit is created with the right keys
        """
        limit = 10
        input_path = self.video_path
        output_path = self.output_path
        mpc = MediaPipeClient(
            video_input_filename="back.mp4",
            video_input_path=input_path,
            video_output_prefix=output_path,
        )
        mpc.process_video(limit=limit)
        mpc.write_pose_data_to_file()

        self.assertEqual(True, os.path.exists(output_path))
        paths = list(Path(output_path).glob("**/*.json"))

        self.assertEqual(len(paths), limit)

        for p in paths:
            with open(str(p), "r") as f:
                pose_data = json.load(f)
                self.assertIn("frame_number", pose_data)
                self.assertIn("joint_positions", pose_data)
                self.assertIn("image_dimensions", pose_data)

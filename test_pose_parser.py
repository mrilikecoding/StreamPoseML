import unittest
import time
from pathlib import Path
import shutil
import os
import json

from pose_parser.pose_parser import MediaPipeClient, MediaPipeClientError


class TestMediaPipeClient(unittest.TestCase):
    def setUp(self) -> None:
        self.video_path = "./test_videos"
        return super().setUp()

    def test_client_init(self):
        """
        GIVEN input/output path and filenames
        WHEN passed into the MediaPipeClient at init
        THEN the client object sets the appropriate output data paths with the current timestamp
        """
        input_path = "./test_videos"
        output_path = "./tmp/data/keypoints"
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
            # cleanup
            shutil.rmtree(mpc.json_output_path)

    def test_client_init_error(self):
        """
        GIVEN no filenames
        WHEN initializing MediaPipeClient
        THEN the client returns an MediaPipe
        """
        self.assertRaises(MediaPipeClientError, lambda: MediaPipeClient())

    def test_process_video(self):
        """
        GIVEN a file passed into a MediaPipeClient
        WHEN process video is called
        THEN pose landmarks are added to the client object
        """
        input_path = "./test_videos"
        output_path = "./tmp/data/keypoints"
        mpc = MediaPipeClient(
            video_input_filename="back.mp4",
            video_input_path=input_path,
            video_output_prefix=output_path,
        )
        mpc.process_video(limit=50)
        self.assertEqual(mpc.frame_count, len(mpc.frame_data_list))
        for pose_data in mpc.frame_data_list:
            self.assertIn("frame_number", pose_data)
            self.assertIn("joint_positions", pose_data)
            self.assertIn("image_dimensions", pose_data)

        # cleanup
        shutil.rmtree(mpc.json_output_path)

    def test_serialize_pose_landmarks(self):
        """
        GIVEN a MediaPipe client and some extracted pose landmarks from a frame
        WHEN the pose landmarks are serialized
        THEN a dictionary is returned containing the appropriate data
        """
        limit = 10
        input_path = "./test_videos"
        output_path = "./tmp/data/keypoints"
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
                    self.assertIn(serialized[joint], "x")
                    self.assertIn(serialized[joint], "y")
                    self.assertIn(serialized[joint], "z")
                    self.assertIn(serialized[joint], "x_normalized")
                    self.assertIn(serialized[joint], "y_normalized")
                    self.assertIn(serialized[joint], "z_normalized")

    def test_write_pose_data_to_file(self):
        """
        GIVEN a MediaPipeClient with a processed video
        WHEN pose data is written to files
        THEN an amount of files equal to the passed limit is created with the right keys
        """
        limit = 10
        input_path = "./test_videos"
        output_path = "./tmp/data/keypoints"
        mpc = MediaPipeClient(
            video_input_filename="back.mp4",
            video_input_path=input_path,
            video_output_prefix=output_path,
        )
        mpc.process_video(limit=limit)
        mpc.write_pose_data_to_file()

        self.assertEqual(True, os.path.exists(output_path))
        paths = Path(output_path).glob("**/*.json")

        self.assertEqual(len(paths))

        for p in paths:
            with open(str(p), "r") as f:
                pose_data = json.load(f)
                self.assertIn("frame_number", pose_data)
                self.assertIn("joint_positions", pose_data)
                self.assertIn("image_dimensions", pose_data)

        # cleanup
        shutil.rmtree(mpc.json_output_path)

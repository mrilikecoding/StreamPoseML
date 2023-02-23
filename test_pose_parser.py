import unittest
import time
from pathlib import Path
import shutil
import os
import json

from pose_parser.pose_parser import (
    MediaPipeClient,
    MediaPipeClientError,
    BlazePoseSequence,
    BlazePoseSequenceError,
    BlazePoseFrame,
    BlazePoseFrameError,
    Joint,
)


class TestAngle(unittest.TestCase):
    pass


class TestJoint(unittest.TestCase):
    pass


class TestBlazePoseFrame(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.output_path = "./tmp/data/keypoints"
        self.video_path = "./test_videos"
        input_path = self.video_path
        output_path = self.output_path
        mpc = MediaPipeClient(
            video_input_filename="back.mp4",
            video_input_path=input_path,
            video_output_prefix=output_path,
        )
        mpc.process_video(limit=50)
        self.bps = BlazePoseSequence(mpc.frame_data_list)

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
        GIVEN a BlazePoseFrame class
        WHEN initialized with a frame of BlazePoseSequence.sequence data
        THEN a BlazePoseFrame object is instantiated regardless of whether joint position data is present
        """
        bpf_no_joint = BlazePoseFrame(frame_data=self.bps.sequence_data[0])
        bpf_joint = BlazePoseFrame(frame_data=self.bps.sequence_data[2])
        self.assertIsInstance(bpf_no_joint, BlazePoseFrame)
        self.assertIsInstance(bpf_joint, BlazePoseFrame)

    def test_set_joint_positions(self):
        """
        GIVEN a BlazePoseFrame class
        WHEN calling to set joint positions when there are joints
        THEN a Joint object is instantiated using the raw joint data on the BlazePoseFrame instance
        """
        self.bps.generate_blaze_pose_frames_from_sequence()
        bpf = self.bps.frames[2]
        bpf.set_joint_positions()
        for joint in bpf.joint_position_names:
            self.assertIsInstance(bpf.joints[joint], Joint)

    def test_generate_angle_measurements(self):
        self.bps.generate_blaze_pose_frames_from_sequence()
        bpf = self.bps.frames[2]
        result = bpf.generate_angle_measurements()
        pass

    def test_validate_joint_position_data(self):
        """
        GIVEN a BlazePoseFrame instance
        WHEN validate joint position data is called with joint positions
        THEN True is returned
        """
        bpf = BlazePoseFrame(frame_data=self.bps.sequence_data[0])
        self.assertEqual(
            bpf.validate_joint_position_data(
                self.bps.sequence_data[2]["joint_positions"]
            ),
            True,
        )

    def test_get_vector(self):
        pass

    def test_get_plumbline_vector(self):
        pass

    def test_serialize_frame_data(self):
        pass

    def test_validate_joint_position_data_invalid(self):
        """
        GIVEN a BlazePoseFrame instance
        WHEN validate joint position data is called with invalid data
        THEN an exception is raised
        """
        bpf = BlazePoseFrame(frame_data=self.bps.sequence_data[0])
        self.assertRaises(
            BlazePoseFrameError, lambda: bpf.validate_joint_position_data({})
        )


class TestBlazePoseSequence(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.output_path = "./tmp/data/keypoints"
        self.video_path = "./test_videos"
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
        bps = BlazePoseSequence(self.frame_data_list)
        self.assertEqual(bps.sequence_data, self.frame_data_list)

    def test_client_init_error(self):
        """
        GIVEN a BlasePoseSequence class
        WHEN initialized with bad data
        THEN a BlazePoseSequenceError is raised
        """
        bad_data = [{"frame_number": 0}]
        self.assertRaises(BlazePoseSequenceError, lambda: BlazePoseSequence((bad_data)))

    def test_validate_pose_schema(self):
        """
        GIVEN a BlazePoseSequence object
        WHEN validate_pose_schema is called
        THEN True is returned if the required keys are present
        """
        bps = BlazePoseSequence(self.frame_data_list)
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
        bps = BlazePoseSequence(self.frame_data_list)
        bad_data = {"frame_number": 0}
        self.assertRaises(
            BlazePoseSequenceError,
            lambda: bps.validate_pose_schema(frame_data=bad_data),
        )

    def test_generate_pose_frames_from_sequence(self):
        bps = BlazePoseSequence(self.frame_data_list)
        bps.generate_blaze_pose_frames_from_sequence()
        self.assertEqual(len(bps.frames), len(self.frame_data_list))
        for frame in bps.frames:
            self.assertIsInstance(frame, BlazePoseFrame)


class TestMediaPipeClient(unittest.TestCase):
    def setUp(self) -> None:
        self.output_path = "./tmp/data/keypoints"
        self.video_path = "./test_videos"
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

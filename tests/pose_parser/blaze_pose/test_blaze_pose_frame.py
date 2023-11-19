import unittest
import shutil


from stream_pose_ml.blaze_pose.mediapipe_client import MediaPipeClient
from stream_pose_ml.blaze_pose.blaze_pose_sequence import BlazePoseSequence
from stream_pose_ml.blaze_pose.blaze_pose_frame import BlazePoseFrame, BlazePoseFrameError
from stream_pose_ml.geometry.joint import Joint
from stream_pose_ml.geometry.vector import Vector
from stream_pose_ml.blaze_pose.openpose_mediapipe_transformer import (
    OpenPoseMediapipeTransformer,
)


class TestBlazePoseFrame(unittest.TestCase):
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
        self.bps = BlazePoseSequence(name="test", sequence=mpc.frame_data_list)

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

    def test_angles_and_distances(self):
        """
        GIVEN a blaze pose frame from joint data
        WHEN angles and distances are calculated
        THEN angles and distances are stored in the corresponding dictionary
        """
        self.bps.generate_blaze_pose_frames_from_sequence()
        bpf = self.bps.frames[2]
        angle_map = OpenPoseMediapipeTransformer().open_pose_angle_definition_map()
        distance_map = (
            OpenPoseMediapipeTransformer().open_pose_distance_definition_map()
        )
        bpf.angles = bpf.generate_angle_measurements(angle_map=angle_map)
        bpf.distances = bpf.generate_distance_measurements(distance_map=distance_map)
        self.assertEqual(True, bool(bpf.angles))
        self.assertEqual(True, bool(bpf.distances))

    def test_generate_angle_measurements(self):
        """
        GIVEN a blaze pose frame from joint data
        WHEN generate angle measurements is called
        THEN angles are calculated and stored in the angles dictionary
        """
        self.bps.generate_blaze_pose_frames_from_sequence()
        bpf = self.bps.frames[2]
        angle_map = OpenPoseMediapipeTransformer().open_pose_angle_definition_map()
        bpf.angles = bpf.generate_angle_measurements(angle_map=angle_map)
        self.assertEqual(True, bool(bpf.angles))
        angle_names = [
            key
            for key in OpenPoseMediapipeTransformer().open_pose_angle_definition_map()
        ]
        for key in bpf.angles.keys():
            self.assertIn(key, angle_names)

    def test_distances_measurements(self):
        """
        GIVEN a blaze pose frame from joint data
        WHEN generate distance measurements is called
        THEN distances are calculated and stored in the distances dictionary
        """
        self.bps.generate_blaze_pose_frames_from_sequence()
        bpf = self.bps.frames[2]
        distance_map = (
            OpenPoseMediapipeTransformer().open_pose_distance_definition_map()
        )
        bpf.distances = bpf.generate_distance_measurements(distance_map=distance_map)
        self.assertEqual(True, bool(bpf.distances))
        distance_names = [
            key
            for key in OpenPoseMediapipeTransformer().open_pose_distance_definition_map()
        ]
        for key in bpf.distances.keys():
            self.assertIn(key, distance_names)

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
        """
        GIVEN a Blaze Pose Frame instance with joints
        WHEN getting a vector between two points
        THEN a Vector object is returned using the coordinates of the two joints
        """
        self.bps.generate_blaze_pose_frames_from_sequence()
        bpf = self.bps.frames[2]
        vector = bpf.get_vector("nose_left_eye", "nose", "left_eye")
        self.assertIsInstance(vector, Vector)
        self.assertEqual("nose_left_eye", vector.name)
        self.assertEqual(bpf.joints["nose"].x, vector.x1)
        self.assertEqual(bpf.joints["left_eye"].x, vector.x2)

    def test_get_average_joint(self):
        """
        GIVEN a Blaze Pose Frame instance with joints
        WHEN getting the average coordinate between two joints
        THEN a Joint object is returned using the coordinates of the average of the two joints
        """
        self.bps.generate_blaze_pose_frames_from_sequence()
        bpf = self.bps.frames[2]
        joint_avg = bpf.get_average_joint("nose_left_eye", "nose", "left_eye")
        self.assertEqual("nose_left_eye", joint_avg.name)
        self.assertIsInstance(joint_avg, Joint)
        self.assertEqual(
            joint_avg.x, (bpf.joints["nose"].x + bpf.joints["left_eye"].x) / 2
        )
        self.assertEqual(
            joint_avg.y, (bpf.joints["nose"].y + bpf.joints["left_eye"].y) / 2
        )
        self.assertEqual(
            joint_avg.z, (bpf.joints["nose"].z + bpf.joints["left_eye"].z) / 2
        )
        self.assertEqual(
            joint_avg.x_normalized,
            (bpf.joints["nose"].x_normalized + bpf.joints["left_eye"].x_normalized) / 2,
        )
        self.assertEqual(
            joint_avg.y_normalized,
            (bpf.joints["nose"].y_normalized + bpf.joints["left_eye"].y_normalized) / 2,
        )
        self.assertEqual(
            joint_avg.z_normalized,
            (bpf.joints["nose"].z_normalized + bpf.joints["left_eye"].z_normalized) / 2,
        )

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

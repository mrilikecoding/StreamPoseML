import os
import time
from pathlib import Path
import json

import mediapipe as mp
import cv2


class BlazePoseFrame:
    """
    This class represents a single frame of BlazePose joint positions
    It stores meta-data related to the frame and also computes angle measurements if joint positions are present
    """

    joint_positions: list
    frame_number: int
    has_joint_positions: bool
    image_dimensions: tuple
    sequence_id: int
    sequence_source: str
    angles: dict

    def __init__(self, frame_data: dict) -> None:
        """
        Initialize this class - passed a dictionary of frame data

        Parameters
        -----
            frame_data: dict
                This is passed from a BlazePoseSequence.sequence_data entry
                ex.
                {
                    'sequence_id': 1677107027968938000,
                    'sequence_source': 'mediapipe',
                    'frame_number': 43,
                    'image_dimensions': {'height': 1080, 'width': 1920},
                    'joint_positions': {
                        'nose': {'x': 12., 'y': 42., 'z': 32., x_normalized: ...}
                        ...
                    }
                }

        """
        self.joint_positions = [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_anle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ]
        self.frame_number = frame_data["frame_number"]
        self.has_joint_positions = bool(frame_data["joint_positions"])
        self.image_dimensions = tuple(frame_data["image_dimensions"])
        self.sequence_id = frame_data["sequence_id"]
        self.sequence_source = frame_data["sequence_source"]
        self.angles = {}
        if self.has_joint_positions:
            self.validate_joint_position_data(frame_data["joint_positions"])
            self.joint_positions = frame_data["joint_positions"]
            self.generate_angle_measurements()

    def validate_joint_position_data(self, joint_positions: dict):
        required_joint_keys = [
            "x",
            "y",
            "z",
            "x_normalized",
            "y_normalized",
            "z_normalized",
        ]

        for joint in self.joint_positions:
            if joint in joint_positions:
                for key in required_joint_keys:
                    if key in joint_positions[joint]:
                        return True
                    else:
                        raise BlazePoseFrameError(
                            f"{key} missing from {joint} position data"
                        )
            else:
                raise BlazePoseFrameError(f"{joint} missing from joint positions dict")

    def generate_angle_measurements(self):
        self.plumb_line_vector = None

    def serialize_frame_data(self):
        pass


class BlazePoseFrameError(Exception):
    """
    Raise when there is an error in the BlazePoseFrame class
    """

    pass


class BlazePoseSequence:
    """
    This class represents a sequence of BlazePoseFrames

    It validates they have the right shape and then creates a BlazePoseFrame for each pass frame

    """

    sequence_data: list  # a list of frame data dicts for keypoints / metadata
    frames: list  # a list of BlazePoseFrames representing the sequence data
    joint_positions: list  # required keys for a non-empty joint position object

    def __init__(self, sequence: list = []) -> None:
        self.joint_positions = [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_anle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ]
        for frame in sequence:
            if not self.validate_pose_schema(frame_data=frame):
                raise BlazePoseSequenceError("Validation error!")

        self.sequence_data = sequence
        self.frames = []

    def validate_pose_schema(self, frame_data: dict):
        """
        This method is responsible for ensuring data meets the required schema

        Parameters
        ------

            frame_data: dict
                a MediaPipeClient.frame_data_list entry conforming to proper schema

        Returns
        -------
            valid: bool
                returns True if the data is valid

        Raises
        _____
            exception: BlazePoseSequenceError
                Raises an exception if there is a problem with validation
        """
        required_keys = [
            "sequence_id",
            "sequence_source",
            "frame_number",
            "image_dimensions",
            "joint_positions",
        ]
        # verify required top level keys are present
        for key in required_keys:
            if key not in frame_data:
                raise BlazePoseSequenceError(
                    f"Validation error - {key} is missing from frame data"
                )

        joint_positions = frame_data["joint_positions"]

        # it is possible there is no joint position data for a frame
        if not joint_positions:
            return True

        # if there is joint position data, ensure all keys are present
        for pos in self.joint_positions:
            if pos not in joint_positions:
                raise BlazePoseSequenceError(
                    f"Validation error - {pos} is missing from joint position data"
                )

        return True

    def generate_blaze_pose_frames_from_sequence(self):
        for frame_data in self.sequence_data:
            bpf = BlazePoseFrame(frame_data=frame_data)
            self.frames.append(bpf)


class BlazePoseSequenceError(Exception):
    """
    Raise when there is an error in the BlazePoseSequence class
    """

    pass


class MediaPipeClient:
    """
    This class provides an interface to Mediapipe for keypoint extraction, sets I/O paths

    See https://google.github.io/mediapipe/solutions/pose.html for information about inner workings of MediaPipe
    """

    frame_count: int
    frame_data_list: list
    video_input_filename: str
    video_input_path: str
    video_output_prefix: str
    id: int
    joints: list  # an ordered list of joints corresponding to MediaPipe BlazePose model

    def __init__(
        self,
        video_input_filename: str = None,
        video_input_path: str = "./test_videos",
        video_output_prefix: str = "./data/keypoints",
        id=int(time.time_ns()),
    ) -> None:
        """
        Client init

        Parameters
        ----
            video_input_filename: str
                the name of the file - "some_file.mp4"
            video_input_path: str
                "path/to/file"
            video_output_prefix: str
                "where/to/put/keypoints"
            id: int
                The id for this client - this will be used to set the output sub-directory

        """
        self._results_raw = []
        self.joints = [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_anle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ]
        self.frame_count = 0
        self.id = id
        # path to OP executable in repo
        self.video_input_path = video_input_path

        self.frame_data_list = []

        if video_input_filename:
            pre = Path(video_input_filename).stem
            self.json_output_path = f"{video_output_prefix}/{pre}-{id}"
            os.makedirs(self.json_output_path)
        else:
            raise MediaPipeClientError("No input file specified")

        self.video_input_filename = video_input_filename

    def process_video(self, limit: int = None):
        """
        This method is responsible for iterating through frames in the input video
        and running the keypoint pose extraction via media pipe.


        See https://github.com/google/mediapipe/issues/1589
        Also see https://google.github.io/mediapipe/solutions/pose.html

        Parameters
        -----
            limit: int
                If a limit is passed in, only process frames up to this number
        """
        # init frame counter
        self.frame_count = 0

        # set up mediapipe
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # start video processing
        cap = cv2.VideoCapture(f"{self.video_input_path}/{self.video_input_filename}")
        if cap.isOpened() == False:
            raise MediaPipeClientError("Error opening file")
        while cap.isOpened():
            # bail if we go over processing limit
            if self.frame_count >= limit:
                return
            ret, image = cap.read()
            if not ret:
                break
            # build data object for this frame
            self.frame_count += 1
            self.image_dimensions = image.shape
            h, w, _ = self.image_dimensions
            frame_data = {
                "sequence_id": self.id,
                "sequence_source": "mediapipe",
                "frame_number": self.frame_count,
                "image_dimensions": {"height": h, "width": w},
                "joint_positions": {},
            }
            # mediapipe does its thing
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # store the pose object for introspection
            self._results_raw.append(results)
            if not results.pose_landmarks:
                self.frame_data_list.append(frame_data)
                continue

            # format pose object how we like
            pose_landmarks = self.serialize_pose_landmarks(
                pose_landmarks=list(results.pose_landmarks.landmark)
            )
            frame_data["joint_positions"] = pose_landmarks
            # add frame to client pose list
            self.frame_data_list.append(frame_data)

    def write_pose_data_to_file(self):
        """
        This method iterates through each pose data dictionary in the pose_data list.
        It then creates a json file at the json output path with all this data
        """
        for frame_data in self.frame_data_list:
            file_path = f"{self.json_output_path}/keypoints-{frame_data['frame_number']:04d}.json"
            with open(file_path, "w") as f:
                json.dump(frame_data, f)
                print(
                    f"Successfully wrote keypoints from {self.video_input_filename} to {f}"
                )

    def serialize_pose_landmarks(self, pose_landmarks: list):
        """
        This method take a list of pose landmarks (casted from the mediapipe pose_landmarks.landmark object)
        and extracts x, y, z data, performs a normalization, then stores all the data in a dictionary

        Note: according to MediaPipe docs "z" uses roughly same scale as x. May not be very accurate.

        Paramters
        -----
            pose_landmarks: list
                Resulting from this process...
                    mp_pose = mp.solutions.pose
                    pose = mp_pose.Pose()
                    pose.process()
                    pose_landmarks = list(results.pose_landmarks.landmark)


        Rerturns
            landmarks: dict
                dictionary containing x, y, z and x_normalized, y_normalized, z_normalized
        """
        landmarks = {}
        if pose_landmarks:
            h, w, _ = self.image_dimensions
            for i, joint in enumerate(self.joints):
                landmarks[joint] = {
                    "x": (pose_landmarks[i].x),
                    "y": (pose_landmarks[i].y),
                    "z": (
                        pose_landmarks[i].z
                    ),  # according to docs, z uses "roughly the same scale as x"
                    "x_normalized": (pose_landmarks[i].x * w),
                    "y_normalized": (pose_landmarks[i].y * h),
                    "z_normalized": (
                        pose_landmarks[i].z * w
                    ),  # according to docs, z uses "roughly the same scale as x"
                }
        return landmarks


class MediaPipeClientError(Exception):
    """Raised when there's an error in this class"""

    pass

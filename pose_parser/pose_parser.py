import os
import time
from pathlib import Path
import json

import mediapipe as mp
import cv2
import numpy as np


class Joint:
    """
    This is a data structure representing the attributes of a joint
    """

    # the name of this joint
    name: str
    # x, y, z as 0-1
    x: float
    y: float
    z: float
    # x, y, z normalized to the image dimensions
    x_normalized: float
    y_normalized: float
    z_normalized: float
    # the image dimensions for the joint's image {"height": 100, "width": 200}
    image_dimensions: dict

    def __init__(self, name, joint_data: dict) -> None:
        """
        This initializes a Joint object with required data

        Parameters
        ---------
            joint_data: dict
                Ex.
                {
                    'image_dimensions': { 'height': 500, 'width': 1000 },
                    'x': 0.7057283520698547,
                    'y': 1.333446979522705,
                    'z': 0.5175799131393433,
                    'x_normalized': 1354.998435974121,
                    'y_normalized': 1440.1227378845215,
                    'z_normalized': 993.7534332275391,
                }
        """
        required_keys = [
            "x",
            "y",
            "z",
            "x_normalized",
            "y_normalized",
            "z_normalized",
            "image_dimensions",
        ]
        if not all([key in required_keys for key in joint_data]):
            raise JointError(
                "The required data is missing from the joint data dictionary."
            )

        self.name = name
        self.image_dimensions = joint_data["image_dimensions"]
        self.x = joint_data["x"]
        self.y = joint_data["y"]
        self.z = joint_data["z"]
        self.x_normalized = joint_data["x_normalized"]
        self.y_normalized = joint_data["y_normalized"]
        self.z_normalized = joint_data["z_normalized"]


class JointError(Exception):
    """Raise when there is an error in the Joint class"""

    pass


class Vector:
    """This is a data structure representing a vector of two joints"""

    name: str  # name of this vector
    x1: float
    y1: float
    z1: float
    x1_normalized: float
    y1_normalized: float
    z1_normalized: float
    x2: float
    y2: float
    z2: float
    x2_normalized: float
    y2_normalized: float
    z2_normalized: float

    def __init__(self, name: str, joint_1: Joint, joint_2: Joint) -> None:
        self.name = name
        self.x1 = joint_1.x
        self.y1 = joint_1.y
        self.z1 = joint_1.z
        self.x1_normalized = joint_1.x_normalized
        self.y1_normalized = joint_1.y_normalized
        self.z1_normalized = joint_1.z_normalized

        self.x2 = joint_2.x
        self.y2 = joint_2.y
        self.z2 = joint_2.z
        self.x2_normalized = joint_2.x_normalized
        self.y2_normalized = joint_2.y_normalized
        self.z2_normalized = joint_2.z_normalized

    def get_coord_tuple(self, normalized: bool = False):
        if normalized:
            return (
                (self.x1_normalized, self.y1_normalized, self.z1_normalized),
                (self.x2_normalized, self.y2_normalized, self.z2_normalized),
            )

        return ((self.x1, self.y1, self.z1), (self.x2, self.y2, self.z2))


class Angle:
    """
    This is a data structure representing angles between 2d and 3d vectors

    From:
    https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249

    """

    name: str
    vector_1: tuple
    vector_2: tuple
    angle_2d: float
    angle_3d: float
    angle_2d_radians: float
    angle_3d_radians: float
    angle_2d_degrees: float
    angle_3d_degrees: float

    def __init__(self, name: str, vector1: Vector, vector2: Vector) -> None:
        """
        Upon initialization this class creates a named angle object
        where the angle between two vectors is calculated in radians
        and degrees with accessible attributes for introspection
        """
        self.name = name
        self.vector_1 = vector1
        self.vector_2 = vector2
        self.angle_2d = self.angle_between(
            vector1.get_coord_tuple()[:2], vector2.get_coord_tuple()[:2]
        )
        self.angle_3d = self.angle_between(
            vector1.get_coord_tuple()[:3], vector2.get_coord_tuple()[:3]
        )
        self.angle_2d_radians = self.angle_2d  # alias
        self.angle_3d_radians = self.angle_3d  # alias
        self.angle_2d_degrees = np.degrees(self.angle_2d)
        self.angle_3d_degrees = np.degrees(self.angle_3d)
        self.angle_2d_normalized = self.angle_between(
            vector1.get_coord_tuple(normalized=True)[:2],
            vector2.get_coord_tuple(normalized=True)[:2],
        )
        self.angle_3d_normalized = self.angle_between(
            vector1.get_coord_tuple(normalized=True)[:3],
            vector2.get_coord_tuple(normalized=True)[:3],
        )
        self.angle_2d_normalized_radians = self.angle_2d  # alias
        self.angle_3d_normalized_radians = self.angle_3d  # alias
        self.angle_2d_normalized_degrees = np.degrees(self.angle_2d)
        self.angle_3d_normalized_degrees = np.degrees(self.angle_3d)

    @staticmethod
    def unit_vector(vector: tuple):
        """Returns the unit vector of the vector."""
        return vector / np.linalg.norm(vector)

    @staticmethod
    def angle_between(self, vector_1: tuple, vector_2: tuple):
        """Returns the angle in radians between vectors 'v1' and 'v2'::

        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793

        """
        v1_u = self.unit_vector(vector_1)
        v2_u = self.unit_vector(vector_2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


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
    joints: dict
    angles: dict
    vectors: dict

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
        self.joint_position_names = [
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
        self.joints = {}
        self.vectors = {}
        self.angles = {}
        self.frame_number = frame_data["frame_number"]
        self.has_joint_positions = bool(frame_data["joint_positions"])
        self.image_dimensions = frame_data["image_dimensions"]
        self.sequence_id = frame_data["sequence_id"]
        self.sequence_source = frame_data["sequence_source"]
        # if we have joint positions, validate them
        # and instantiate joint objects into dictionary
        if self.has_joint_positions:
            self.validate_joint_position_data(frame_data["joint_positions"])
            self.joint_positions_raw = frame_data["joint_positions"]
            self.joints = self.set_joint_positions()
            self.angles = self.generate_angle_measurements()

    def set_joint_positions(self) -> dict:
        """
        This method takes the raw joint data from every named joint
        and formats a data object to create a Joint object instance

        Returns

            joint_positions: dict
                A joint position dictionary where each key is the name of
                a joint and the value is a dictionary containing position
                data for that joint in this frame instance
        """
        if not self.has_joint_positions:
            raise BlazePoseFrameError("There are no joint positions to set")
        joint_positions = {}
        for joint in self.joint_position_names:
            joints_raw = self.joint_positions_raw
            joint_data = {
                "image_dimensions": self.image_dimensions,
                "x": joints_raw[joint]["x"],
                "y": joints_raw[joint]["y"],
                "z": joints_raw[joint]["z"],
                "x_normalized": joints_raw[joint]["x_normalized"],
                "y_normalized": joints_raw[joint]["y_normalized"],
                "z_normalized": joints_raw[joint]["z_normalized"],
            }
            joint_positions[joint] = Joint(name=joint, joint_data=joint_data)
        return joint_positions

    def validate_joint_position_data(self, joint_positions: dict):
        """
        This method validates that the required keys are present in
        the joint position data

        Parameters
        --------
            joint_positions: dict
                a dictionary of joint position data to be validated

        Returns
        ______
            success: bool
                If all keys are present return true

        Raise
        -----
            BlazePoseFrameError if we are missing a key


        """
        required_joint_keys = [
            "x",
            "y",
            "z",
            "x_normalized",
            "y_normalized",
            "z_normalized",
        ]

        for joint in self.joint_position_names:
            if joint in joint_positions:
                for key in required_joint_keys:
                    if key in joint_positions[joint]:
                        continue
                    else:
                        raise BlazePoseFrameError(
                            f"{key} missing from {joint} position data"
                        )
            else:
                raise BlazePoseFrameError(f"{joint} missing from joint positions dict")

        return True

    def generate_angle_measurements(self):
        if not self.has_joint_positions:
            raise BlazePoseFrameError(
                f"There are no joint data to generate angles from"
            )

        # compute plumb line vector as the angle basis
        self.joints["neck"] = self.get_average_joint(
            name="neck", joint_1="left_shoulder", joint_2="right_shoulder"
        )
        self.joints["mid_hip"] = self.get_average_joint(
            name="mid_hip", joint_1="left_hip", joint_2="right_hip"
        )
        self.vectors["plumb_line"] = self.get_vector("neck", "mid_hip")

    def get_vector(self, name: str, joint_name_1: str, joint_name_2: str):
        return Vector(name, self.joints[joint_name_1], self.joints[joint_name_2])

    def get_average_joint(self, name: str, joint_1: str, joint_2: str):
        """
        Blaze pose has a specific set of joints. However, some instances we want
        to compute the value of a mid point. In these cases take the average of two
        named joints

        Parameters
        ---------
            name: str
                what to name the joint that is generated
            joint_1: str
                the name of the joint to lookup in self.joints
            joint_2: str
                the name of the joint to lookup in self.joints

        Returns
        -------
            joint: Joint
                a Joint object representing a joint at the average of the passed joints
        """
        x = (self.joints[joint_1].x + self.joints[joint_2].x) / 2
        y = (self.joints[joint_1].y + self.joints[joint_2].y) / 2
        z = (self.joints[joint_1].z + self.joints[joint_2].z) / 2
        x_normalized = (
            self.joints[joint_1].x_normalized + self.joints[joint_2].x_normalized
        ) / 2
        y_normalized = (
            self.joints[joint_1].y_normalized + self.joints[joint_2].y_normalized
        ) / 2
        z_normalized = (
            self.joints[joint_1].z_normalized + self.joints[joint_2].z_normalized
        ) / 2

        return Joint(
            name=name,
            joint_data={
                "x": x,
                "y": y,
                "z": z,
                "x_normalized": x_normalized,
                "y_normalized": y_normalized,
                "z_normalized": z_normalized,
                "image_dimensions": self.image_dimensions,
            },
        )

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

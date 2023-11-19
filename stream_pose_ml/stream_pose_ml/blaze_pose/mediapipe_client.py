import os
import time
import numpy as np
from pathlib import Path
import json
import cv2
import mediapipe as mp

from stream_pose_ml.blaze_pose.enumerations import BlazePoseJoints


class MediaPipeClient:
    """This class provides an interface to Mediapipe for keypoint extraction, sets I/O paths.

    See https://google.github.io/mediapipe/solutions/pose.html for information about inner workings of MediaPipe
    """

    frame_count: int
    frame_data_list: list
    video_input_filename: str
    video_input_path: str
    video_output_prefix: str
    id: int
    joints: list  # an ordered list of joints corresponding to MediaPipe BlazePose model
    configuration: dict  # options to pass into mediapipe pose
    preprocess_video: bool  # whether to preprocess the video frame

    def __init__(
        self,
        video_input_filename: str = None,
        video_input_path: str = ".stream_pose_ml/test_videos",
        video_output_prefix: str = ".tmp/data/keypoints",
        id: int = int(time.time_ns()),
        configuration: dict = {},
        preprocess_video: bool = False,
        dummy_client: bool = False,
    ) -> None:
        """Initalize a mediapipe client object.

        Args:
            video_input_filename: str
                the name of the file - "some_file.mp4"
            video_input_path: str
                "path/to/file"
            video_output_prefix: str
                "where/to/put/keypoints"
            id: int
                The id for this client - this will be used to set the output sub-directory
            dummy_client: bool
                If true, no input file is needed. Use this when only calling static methods

        """
        self.configuration = configuration
        self.preprocess_video = preprocess_video
        self._results_raw = []
        self.joints = [joint.name for joint in BlazePoseJoints]
        self.frame_count = 0
        self.id = id
        # path to OP executable in repo
        self.video_output_prefix = video_output_prefix
        self.video_input_path = video_input_path
        self.video_input_filename = video_input_filename
        self.frame_data_list = []

        if video_input_filename:
            self.video_input_filename = video_input_filename
            pre = Path(self.video_input_filename).stem
            self.json_output_path = f"{self.video_output_prefix}/{pre}-{id}"
        elif not dummy_client:
            raise MediaPipeClientError("No input file specified")

    def import_mediapipe(self) -> None:
        if self.dummy_client:
            return
        else:
            import mediapipe as mp

    def process_video(self, limit: int = None) -> "MediaPipeClient":
        """This method runs mediapipe on a video referenced by this object.

        This method is responsible for iterating through frames in the input video
        and running the keypoint pose extraction via media pipe.


        See https://github.com/google/mediapipe/issues/1589
        Also see https://google.github.io/mediapipe/solutions/pose.html

        Args:
            limit: int
                If a limit is passed in, only process frames up to this number

        Returns:
            self: MediaPipeClient
                returns this instance for chaining to init
        """
        # init frame counter
        self.frame_count = 0

        # set up mediapipe
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # start video processing
        cap = cv2.VideoCapture(f"{self.video_input_path}/{self.video_input_filename}")
        if cap.isOpened() == False:
            raise MediaPipeClientError(
                f"Error opening file: {self.video_input_path}/{self.video_input_filename}"
            )
        while cap.isOpened():
            # TODO refactor this into stages
            # bail if we go over processing limit
            if limit is not None and self.frame_count >= limit:
                return
            ret, image = cap.read()
            if not ret:
                break
            if self.preprocess_video:
                image = self.run_preprocess_video(image=image)
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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.process_frame(image, pose)
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
        return self

    @staticmethod
    def process_frame(image, pose):
        # set up mediapipe
        return pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def write_pose_data_to_file(self):
        """Write this object's video pose data to file.

        This method iterates through each pose data dictionary in the pose_data list.
        It then creates a json file at the json output path with all this data
        """
        try:
            os.makedirs(self.json_output_path)
            for frame_data in self.frame_data_list:
                file_path = f"{self.json_output_path}/keypoints-{frame_data['frame_number']:04d}.json"
                with open(file_path, "w") as f:
                    json.dump(frame_data, f)
                    # print(
                    #     f"Successfully wrote keypoints from {self.video_input_filename} to {file_path}"
                    # )
        except:
            raise MediaPipeClientError("There was a problem writing pose data to json")

    @staticmethod
    def run_preprocess_video(image: np.ndarray) -> np.ndarray:
        """Run some basic image preprocessing steps.

        Enhance contrast

        Args:
            image: np.ndarray
                an image
        Returns:
            the image run through preprocessing steps

        """
        # Denoising
        # denoised_img = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # Contrast Enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_img = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)

        return enhanced_img

    @staticmethod
    def get_joint_coordinates(
        joints: list[str], reference_joint_name: str, pose_landmarks: list
    ) -> list[float]:
        """
        Get the x and y coordinates of the specified joint.

        Args:
            joints: List of joint names.
            reference_joint_name: Name of the reference joint.
            pose_landmarks: List of pose landmarks.

        Returns:
            List containing the x and y coordinates of the specified joint.
        """
        joint_index = joints.index(reference_joint_name)
        try:
            return [pose_landmarks[joint_index].x, pose_landmarks[joint_index].y]
        except:
            return [pose_landmarks[joint_index]["x"], pose_landmarks[joint_index]["y"]]

    @staticmethod
    def calculate_reference_point_distance(
        joint_1: list[float], joint_2: list[float]
    ) -> float:
        """
        Calculate the Euclidean distance between two joint coordinates.

        Args:
            joint_1: List containing the x and y coordinates of the first joint.
            joint_2: List containing the x and y coordinates of the second joint.

        Returns:
            Euclidean distance between the two joint coordinates.
        """
        return np.linalg.norm(np.array(joint_1) - np.array(joint_2))

    @staticmethod
    def calculate_reference_point_midpoint(
        joint_1: list[float], joint_2: list[float]
    ) -> dict[str, float]:
        """
        Calculate the midpoint of two joint coordinates.

        Args:
            joint_1: List containing the x and y coordinates of the first joint.
            joint_2: List containing the x and y coordinates of the second joint.

        Returns:
            Dictionary containing the x and y coordinates of the midpoint.
        """
        return {"x": (joint_1[0] + joint_2[0]) / 2, "y": (joint_1[1] + joint_2[1]) / 2}

    def serialize_pose_landmarks(self, pose_landmarks: list):
        """Get a formatted list of video data coordinates.

        This method take a list of pose landmarks (casted from the mediapipe pose_landmarks.landmark object)
        and extracts x, y, z data, performs a normalization of reference joints, then stores all the data in a dictionary

        Note: according to MediaPipe docs "z" uses roughly same scale as x. May not be very accurate.

        Args:
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
            # TODO @mrilikecoding pass the reference joints in
            reference_joint_1 = "left_hip"
            reference_joint_2 = "right_hip"
            joint_1_coordinates = self.get_joint_coordinates(
                self.joints, reference_joint_1, pose_landmarks
            )
            joint_2_coordinates = self.get_joint_coordinates(
                self.joints, reference_joint_2, pose_landmarks
            )
            reference_point_distance = self.calculate_reference_point_distance(
                joint_1_coordinates, joint_2_coordinates
            )
            reference_point_midpoint = self.calculate_reference_point_midpoint(
                joint_1_coordinates, joint_2_coordinates
            )
            for i, joint in enumerate(self.joints):
                try:
                    x_normed = (
                        pose_landmarks[i].x / reference_point_distance
                    ) - reference_point_midpoint["x"]
                    y_normed = (
                        pose_landmarks[i].y / reference_point_distance
                    ) - reference_point_midpoint["y"]
                    # Not normalizing z here, as the coordinate is not accurate
                    # according to docs, z uses "roughly the same scale as x"
                    landmarks[joint] = {
                        "x": pose_landmarks[i].x,
                        "y": pose_landmarks[i].y,
                        "z": pose_landmarks[i].z,
                        "x_normalized": x_normed,
                        "y_normalized": y_normed,
                        "z_normalized": pose_landmarks[i].z,
                    }
                except:
                    x_normed = (
                        pose_landmarks[i]["x"] / reference_point_distance
                    ) - reference_point_midpoint["x"]
                    y_normed = (
                        pose_landmarks[i]["y"] / reference_point_distance
                    ) - reference_point_midpoint["y"]
                    # Not normalizing z here, as the coordinate is not accurate
                    # according to docs, z uses "roughly the same scale as x"
                    landmarks[joint] = {
                        "x": pose_landmarks[i]["x"],
                        "y": pose_landmarks[i]["y"],
                        "z": pose_landmarks[i]["z"],
                        "x_normalized": x_normed,
                        "y_normalized": y_normed,
                        "z_normalized": pose_landmarks[i]["z"],
                    }
        return landmarks


class MediaPipeClientError(Exception):
    """Raised when there's an error in this class"""

    pass

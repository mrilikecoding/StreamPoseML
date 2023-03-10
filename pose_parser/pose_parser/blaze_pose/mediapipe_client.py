import os
import time
from pathlib import Path
import json
import mediapipe as mp
import cv2

from pose_parser.blaze_pose.enumerations import BlazePoseJoints


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

    def __init__(
        self,
        video_input_filename: str = None,
        video_input_path: str = ".pose_parser/test_videos",
        video_output_prefix: str = ".tmp/data/keypoints",
        id=int(time.time_ns()),
        configuration={},
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

        """
        self.configuration = {}
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
        else:
            raise MediaPipeClientError("No input file specified")

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
            # bail if we go over processing limit
            if limit and self.frame_count >= limit:
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
        return self

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
                    print(
                        f"Successfully wrote keypoints from {self.video_input_filename} to {file_path}"
                    )
        except:
            raise MediaPipeClientError("There was a problem writing pose data to json")

    def serialize_pose_landmarks(self, pose_landmarks: list):
        """Get a formatted list of video data coordinates.

        This method take a list of pose landmarks (casted from the mediapipe pose_landmarks.landmark object)
        and extracts x, y, z data, performs a normalization, then stores all the data in a dictionary

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

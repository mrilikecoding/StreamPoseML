import base64
import time
import typing
from collections import deque
from typing import Union

import cv2  # type: ignore[import-untyped]
import mediapipe as mp  # type: ignore[import-untyped]
import numpy as np

from stream_pose_ml.blaze_pose.blaze_pose_sequence import BlazePoseSequence

from .serializers.blaze_pose_sequence_serializer import (
    BlazePoseSequenceSerializer,
)

if typing.TYPE_CHECKING:
    from stream_pose_ml.blaze_pose.mediapipe_client import MediaPipeClient
    from stream_pose_ml.transformers.sequence_transformer import SequenceTransformer

    from .learning.trained_model import TrainedModel


class StreamPoseClient:
    def __init__(
        self,
        frame_window: int = 25,
        mediapipe_client_instance: Union["MediaPipeClient", None] = None,
        trained_model: Union["TrainedModel", None] = None,
        data_transformer: Union["SequenceTransformer", None] = None,
    ):
        self.frame_window = frame_window
        self.model = trained_model
        self.transformer = data_transformer
        self.mpc = mediapipe_client_instance
        self.frames: deque[typing.Any] = deque([], maxlen=self.frame_window)
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.current_classification: bool | None = None

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Run some basic image preprocessing steps. Currently just contrast enhance

        Enhance contrast

        Args:
            image: np.ndarray
                an image
        Returns: np.ndarray
            the image run through preprocessing steps

        """
        # Contrast Enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(lightness)
        enhanced_img = cv2.merge([cl, a, b])  # type: ignore[arg-type]
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)

        return enhanced_img

    def run_keypoint_pipeline(self, keypoints):
        current_frames = self.update_frame_data_from_js_client_keypoints(keypoints)
        if len(current_frames) == self.frame_window:
            if self.model is None or self.transformer is None:
                return False

            sequence = BlazePoseSequence(
                name=f"sequence-{time.time_ns()}",
                sequence=list(current_frames),
                include_geometry=True,
            ).generate_blaze_pose_frames_from_sequence()
            sequence_data = BlazePoseSequenceSerializer().serialize(sequence)
            # TODO why is the target showing up in the 'columns' array here?
            # Pulling off X_test instead...
            columns = self.model.model_data["X_test"].columns.tolist()
            data, meta = self.transformer.transform(data=sequence_data, columns=columns)
            self.current_classification = bool(self.model.predict(data=data)[0])
        return True

    def run_frame_pipeline(self, image: np.ndarray):
        results = self.get_keypoints(image)
        current_frames = self.update_frame_data(results)
        if len(current_frames) == self.frame_window:
            if self.model is None or self.transformer is None:
                return False

            sequence = BlazePoseSequence(
                name=f"sequence-{time.time_ns()}",
                sequence=list(current_frames),
                include_geometry=True,
            ).generate_blaze_pose_frames_from_sequence()
            sequence_data = BlazePoseSequenceSerializer().serialize(sequence)
            # TODO why is the target showing up in the 'columns' array here?
            # Pulling off X_test instead...
            columns = self.model.model_data["X_test"].columns.tolist()
            data, meta = self.transformer.transform(data=sequence_data, columns=columns)
            self.current_classification = bool(self.model.predict(data=data)[0])
        return True

    def update_frame_data_from_js_client_keypoints(self, keypoint_results):
        frame_data = {
            "sequence_id": None,
            "sequence_source": "web",
            "frame_number": None,
            "image_dimensions": None,
        }
        if "landmarks" in keypoint_results and len(keypoint_results["landmarks"]):
            frame_data["joint_positions"] = self.mpc.serialize_pose_landmarks(
                pose_landmarks=keypoint_results["landmarks"][0]
            )

            self.frames.append(frame_data)
        return self.frames

    def update_frame_data(self, keypoint_results):
        frame_data = {
            "sequence_id": None,
            "sequence_source": "web",
            "frame_number": None,
            "image_dimensions": None,
        }
        if keypoint_results:
            frame_data["joint_positions"] = self.mpc.serialize_pose_landmarks(
                pose_landmarks=list(keypoint_results.pose_landmarks.landmark)
            )
            self.frames.append(frame_data)
        return self.frames

    def get_keypoints(self, image: np.ndarray):
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return results

    @staticmethod
    def convert_base64_to_image_array(base64_string: str) -> np.ndarray:
        """Convert a base64 image string into a numpy array

        Args:
            decoded_image_data: bytes
                a decoded base64 image
        Returns: np.ndarray
            an numpy array representation of a color image
        """
        # Decode the base64 encoded frame data to an image
        decoded_image_data = base64.b64decode(base64_string.split(",")[1])
        nparr = np.frombuffer(decoded_image_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

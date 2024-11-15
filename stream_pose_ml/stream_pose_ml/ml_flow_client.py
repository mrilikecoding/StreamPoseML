from threading import current_thread
import time
import typing

from collections import deque
from stream_pose_ml.blaze_pose.blaze_pose_sequence import BlazePoseSequence
from stream_pose_ml.services import segmentation_service as ss
from stream_pose_ml.serializers.blaze_pose_sequence_serializer import (
    BlazePoseSequenceSerializer,
)

if typing.TYPE_CHECKING:
    from stream_pose_ml.blaze_pose.mediapipe_client import MediaPipeClient
    from stream_pose_ml.learning.trained_model import TrainedModel
    from stream_pose_ml.transformers.sequence_transformer import SequenceTransformer


class MLFlowClient:
    def __init__(
        self,
        mediapipe_client_instance: type["MediaPipeClient"] | None = None,
        trained_model: type["TrainedModel"] | None = None,
        data_transformer: type["SequenceTransformer"] | None = None,
        predict_fn: typing.Callable | None = None,
        input_example: dict = {"columns": []},
        frame_window: int = 30,
        frame_overlap: int = 5,
    ):
        self.frame_window = frame_window
        self.frame_overlap = frame_overlap
        self.model = trained_model
        self.transformer = data_transformer
        self.mpc = mediapipe_client_instance
        self.frames = deque([], maxlen=self.frame_window)
        self.current_classification = None
        self.predict_fn = predict_fn
        self.input_example = input_example
        self.input_example_columns = input_example["columns"]
        self.update_frame_frequency = self.frame_window - self.frame_overlap
        self.counter = 0
        self.prediction_processing_time = None
        self.last_prediction_timestamp = 0

    def run_keypoint_pipeline(self, keypoints):
        current_frames = self.update_frame_data_from_js_client_keypoints(keypoints)
        self.counter += 1
        if (
            len(current_frames) == self.frame_window
            and self.counter >= self.update_frame_frequency
        ):
            self.counter = 0
            sequence = BlazePoseSequence(
                name=f"sequence-{time.time_ns()}",
                sequence=list(current_frames),
                include_geometry=False,
            ).generate_blaze_pose_frames_from_sequence()

            sequence_data = BlazePoseSequenceSerializer().serialize(sequence)
            data, meta = self.transformer.transform(
                data=sequence_data, columns=self.input_example_columns
            )
            if not self.predict_fn:
                return

            # TODO enforce signature of predict_fn, this is brittle
            start_time = time.time()
            prediction = self.predict_fn(json_data_payload=data)["predictions"][0]
            current_time = time.time()
            speed = current_time - start_time
            self.prediction_processing_time = speed
            self.last_prediction_timestamp = current_time
            self.current_classification = bool(prediction)
            return True
        return False

    def update_frame_data_from_js_client_keypoints(self, keypoint_results):
        frame_data = {
            "sequence_id": None,
            "sequence_source": "web",
            "frame_number": None,
            "image_dimensions": None,
        }
        if "landmarks" in keypoint_results and len(keypoint_results["landmarks"]):
            # mpc.serialize_pose_landmarks takes the coords and formats them with
            # x y z as well as normalized againt the hip width distance
            frame_data["joint_positions"] = self.mpc.serialize_pose_landmarks(
                pose_landmarks=keypoint_results["landmarks"][0]
            )

            self.frames.append(frame_data)

        return self.frames

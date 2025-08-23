import logging
import time
import typing
from collections import deque

from stream_pose_ml.blaze_pose.blaze_pose_sequence import BlazePoseSequence

logger = logging.getLogger(__name__)

from .serializers.blaze_pose_sequence_serializer import (
    BlazePoseSequenceSerializer,
)

if typing.TYPE_CHECKING:
    from stream_pose_ml.blaze_pose.mediapipe_client import MediaPipeClient
    from stream_pose_ml.transformers.sequence_transformer import SequenceTransformer

    from .learning.trained_model import TrainedModel


class MLFlowClient:
    def __init__(
        self,
        mediapipe_client_instance: type["MediaPipeClient"] | None = None,
        trained_model: type["TrainedModel"] | None = None,
        data_transformer: type["SequenceTransformer"] | None = None,
        predict_fn: typing.Callable | None = None,
        input_example: dict | None = None,
        frame_window: int = 30,
        frame_overlap: int = 5,
    ):
        if input_example is None:
            input_example = {"columns": []}
        self.frame_window = frame_window
        self.frame_overlap = frame_overlap
        self.model = trained_model
        self.transformer = data_transformer
        self.mpc = mediapipe_client_instance
        self.frames: deque = deque([], maxlen=self.frame_window)
        self.current_classification = None
        self.predict_fn = predict_fn
        self.input_example = input_example
        self.input_example_columns = input_example["columns"]
        self.update_frame_frequency = self.frame_window - self.frame_overlap
        self.counter = 0
        self.prediction_processing_time = None
        self.last_prediction_timestamp = 0

    def run_keypoint_pipeline(self, keypoints):
        # Track when frame arrives
        frame_arrival_time = time.time()
        
        current_frames = self.update_frame_data_from_js_client_keypoints(keypoints)
        self.counter += 1
        
        # Log frame timing
        time_since_last = frame_arrival_time - self.last_prediction_timestamp
        logger.debug(f"Frame {self.counter}/{self.update_frame_frequency}, "
                    f"time since last prediction: {time_since_last:.3f}s")
        
        if (
            len(current_frames) == self.frame_window
            and self.counter >= self.update_frame_frequency
        ):
            # Rate limiting: Don't classify more than once per 0.8 seconds
            MIN_CLASSIFICATION_INTERVAL = 0.8
            if time_since_last < MIN_CLASSIFICATION_INTERVAL:
                logger.debug(f"Skipping classification - too soon ({time_since_last:.3f}s < {MIN_CLASSIFICATION_INTERVAL}s)")
                # Don't reset counter when rate limited!
                return False
            
            # Log that we're triggering a classification
            logger.info(f"Triggering classification at counter={self.update_frame_frequency}, "
                       f"time since last: {time_since_last:.3f}s")
            self.counter = 0
            
            # Update timestamp BEFORE doing the inference (not after)
            self.last_prediction_timestamp = frame_arrival_time
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
            logger.debug(f"Starting MLflow inference...")
            
            try:
                response = self.predict_fn(json_data_payload=data)
                
                # Handle different response formats
                if isinstance(response, dict):
                    if "predictions" in response:
                        prediction = response["predictions"][0]
                    elif "prediction" in response:
                        prediction = response["prediction"]
                    elif isinstance(response, list):
                        prediction = response[0]
                    else:
                        # Log the response structure for debugging
                        logger.error(f"Unexpected MLflow response format: {response.keys()}")
                        prediction = list(response.values())[0]
                else:
                    prediction = response[0] if isinstance(response, list) else response
                    
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Error parsing MLflow response: {e}")
                logger.error(f"Response was: {response}")
                return False
                
            current_time = time.time()
            speed = current_time - start_time
            self.prediction_processing_time = speed
            # Note: last_prediction_timestamp already set BEFORE inference to prevent burst triggers
            self.current_classification = bool(prediction)
            
            # Log classification result and timing
            logger.info(f"Classification complete: {self.current_classification}, "
                       f"inference time: {speed:.3f}s")
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

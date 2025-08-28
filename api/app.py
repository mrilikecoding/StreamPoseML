import json
import logging
import os

# Import the core module and its components
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import requests
from engineio.payload import Payload  # type: ignore[import-untyped]
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

# Add the root directory to the Python path (needed for imports)
root_dir = str(Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Now import the modules
# ruff: noqa: E402
from stream_pose_ml.blaze_pose.mediapipe_client import MediaPipeClient
from stream_pose_ml.learning.model_builder import ModelBuilder
from stream_pose_ml.learning.trained_model import TrainedModel
from stream_pose_ml.ml_flow_client import MLFlowClient
from stream_pose_ml.stream_pose_client import StreamPoseClient
from stream_pose_ml.transformers.sequence_transformer import (
    MLFlowTransformer,
    TenFrameFlatColumnAngleTransformer,
)

### Set the model ###
mb = ModelBuilder()
model_location = "./data/trained_models"
trained_model_instance = TrainedModel()


### Set the pose estimation client ###
# dummy_client=True here indicates that we don't need mediapipe loaded,
# we just want to use methods on the class
mpc = MediaPipeClient(dummy_client=True)


### Create StreamPoseML client wrapper to instantiate from front end ###
class StreamPoseMLApp:
    def __init__(self):
        self.stream_pose_client = None

    def set_stream_pose_client(self, stream_pose_client):
        self.stream_pose_client = stream_pose_client

    def set_actuator(self, actuator="bluetooth_device"):
        # TODO this is deprecated as we're calling bluetooth from the front end
        if actuator == "bluetooth_device":
            # self.actuator = BluetoothDevice()  # TODO: Import or remove
            pass

    def actuate(self, data):
        self.actuator.send(data)
        return self.actuator.receive()


stream_pose = StreamPoseMLApp()

### Init Flask API ###
app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max file size

# WebSocket configuration - Reduced buffer to prevent excessive queuing
# At 30fps, 90 packets = 3 seconds of buffer
WEBSOCKET_BUFFER_SIZE = 90
Payload.max_decode_packets = WEBSOCKET_BUFFER_SIZE
socketio: SocketIO = SocketIO(
    app,
    async_mode="eventlet",
    ping_timeout=30,
    ping_interval=20,
    cors_allowed_origins="*",
)

# Tmp file upload - use absolute path to ensure we use the shared volume
# Check if running in Docker container (the /usr/src/app path exists)
if os.path.exists("/usr/src/app"):
    UPLOAD_FOLDER = "/usr/src/app/tmp"
else:
    # Local development
    UPLOAD_FOLDER = os.path.abspath("./tmp")

ALLOWED_EXTENSIONS = {"zip", "tar.gz", "tar", "pickle", "joblib", "model"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)


# TODO - make env dependent from config
whitelist = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://web_ui:3000",
    "http://localhost:5001",
    "http://127.0.0.1:5001",
    "http://stream_pose_ml:5001",
    "https://cdn.jsdelivr.net",
]
CORS(
    app,
    origins=whitelist,
    supports_credentials=True,
    expose_headers=["Content-Type", "Authorization"],
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
)
# CORS(app, origins="*")


def allowed_file(filename):
    # Special handling for tar.gz files
    if filename.lower().endswith(".tar.gz"):
        return "tar.gz" in ALLOWED_EXTENSIONS
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def status():
    return "Server Ready"


### Application Routes ###
@app.route("/set_model", methods=["POST", "OPTIONS"])
def set_model():
    # Handle OPTIONS request for CORS
    if request.method == "OPTIONS":
        return "", 200

    frame_window = request.form.get("frame_window_size", type=int)
    frame_overlap = request.form.get("frame_window_overlap", type=int)

    if "file" not in request.files:
        return jsonify({"result": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"result": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Ensure we use absolute path
        upload_folder = os.path.abspath(app.config["UPLOAD_FOLDER"])
        os.makedirs(upload_folder, exist_ok=True)
        model_path = os.path.join(upload_folder, filename)
        file.save(model_path)

        # Extract the archive
        extract_to = os.path.join(upload_folder, filename.rsplit(".", 1)[0])

        # Handle different archive formats
        try:
            if filename.endswith(".zip"):
                with zipfile.ZipFile(model_path, "r") as zip_ref:
                    zip_ref.extractall(extract_to)
                model_path = extract_to
            elif filename.endswith(".tar.gz") or filename.endswith(".tar"):
                with tarfile.open(model_path, "r:*") as tar_ref:
                    tar_ref.extractall(extract_to)
                model_path = extract_to
        except (zipfile.BadZipFile, tarfile.ReadError) as e:
            print(f"[ERROR] Failed to extract archive: {e}")
            return jsonify({"result": f"Invalid archive file: {str(e)}"}), 400
        except Exception as e:
            print(f"[ERROR] Unexpected error extracting file: {e}")
            return jsonify({"result": f"Error processing file: {str(e)}"}), 500

        # Send a request to mlflow to load the model
        mlflow_response = load_model_in_mlflow(model_path)

        # Check if MLFlow loading was successful
        if mlflow_response is True:
            # Load input_example.json if it exists
            input_example_path = os.path.join(model_path, "input_example.json")
            input_example = None
            if os.path.exists(input_example_path):
                with open(input_example_path) as json_file:
                    input_example = json.load(json_file)
            set_ml_flow_client(
                input_example=input_example,
                frame_window=frame_window,
                frame_overlap=frame_overlap,
            )
            logging.info("MLFlow Model loaded successfully")
            return (
                jsonify({"result": f"MLFlow Ready: classifier set to {filename}."}),
                200,
            )
        elif mlflow_response is False:
            # MLFlow service is not available, fall back to StreamPoseML
            set_stream_pose_ml_client()
            logging.info("StreamPoseML Model loaded successfully")
            return (
                jsonify(
                    {"result": f"StreamPoseML Ready: classifier set to {filename}."}
                ),
                200,
            )
        else:
            # MLFlow returned an error response, fall back to StreamPoseML
            set_stream_pose_ml_client()
            logging.info("Falling back to StreamPoseML after MLFlow error")
            return (
                jsonify(
                    {"result": f"StreamPoseML Ready: classifier set to {filename}."}
                ),
                200,
            )
    else:
        print("invalid file type")
        return jsonify({"result": "Invalid file type"}), 400


def set_stream_pose_ml_client():
    transformer = TenFrameFlatColumnAngleTransformer()
    trained_model_instance.set_data_transformer(transformer)

    stream_pose.set_stream_pose_client(
        StreamPoseClient(
            mediapipe_client_instance=mpc,
            trained_model=trained_model_instance,
            data_transformer=transformer,
            frame_window=10,  # TODO receive from UI
        )
    )


def mlflow_predict(json_data_payload: str):
    # Convert the DataFrame to a single row represented as a list of lists
    data_json = {"inputs": json_data_payload}
    headers = {"Content-Type": "application/json"}

    # Send the request to the /invocations endpoint
    response = requests.post(
        "http://mlflow:5002/invocations", json=data_json, headers=headers
    )

    # Return the response in a usable format
    if response.status_code == 200:
        return response.json()
    else:
        return {"status": "error", "message": response.content}


def set_ml_flow_client(input_example=None, frame_window=30, frame_overlap=5):
    transformer = MLFlowTransformer()
    trained_model_instance.set_data_transformer(transformer)

    stream_pose.set_stream_pose_client(
        MLFlowClient(
            mediapipe_client_instance=mpc,
            trained_model=trained_model_instance,
            data_transformer=transformer,
            frame_window=frame_window,
            predict_fn=mlflow_predict,
            input_example=input_example,
            frame_overlap=frame_overlap,
        )
    )


### SocketIO Listeners ###


def check_connection_health():
    """Periodic check to detect and recover from stuck connections"""
    global last_successful_emit, emit_failures, connection_issues_detected

    current_time = time.time()
    time_since_emit = (
        current_time - last_successful_emit if last_successful_emit > 0 else 0
    )

    if time_since_emit > 30:  # No successful emit in 30 seconds
        print(f"[WARNING] No successful emit in {time_since_emit:.1f} seconds")

        # Try to send a recovery signal
        try:
            socketio.emit(
                "connection_check", {"status": "checking", "timestamp": current_time}
            )
            print("[INFO] Sent connection check signal")
        except Exception as e:
            print(f"[ERROR] Connection check failed: {e}")
            connection_issues_detected += 1

            # Force clear any stuck buffers by restarting the eventlet server
            if time_since_emit > 60:
                print(
                    "[CRITICAL] Connection dead for 60+ seconds, "
                    "may need manual restart"
                )
                # Could implement auto-restart logic here if needed


def load_model_in_mlflow(model_path):
    # The path needs to be adjusted for the mlflow container
    # Since '/usr/src/app/tmp' in 'stream_pose_ml_api' -> '/models' in 'mlflow'

    # Convert the API container path to MLFlow container path
    mlflow_model_path = model_path.replace("/usr/src/app/tmp", "/models")

    data = {"model_path": mlflow_model_path}
    try:
        response = requests.post("http://mlflow:5002/load_model", json=data)
        if response.status_code != 200:
            print(f"[ERROR] MLFlow failed to load model: {response.text}")
            return response  # Return the response object for error handling
        return True  # Success
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Error loading model in mlflow: {e}")
        return False


# Track metrics for performance monitoring
frame_counter = 0
frames_dropped = 0
last_emit_time: float = 0.0
classifications_completed = 0
classification_times = []  # Store last 10 classification timestamps
start_time = time.time()  # Server start time

# WebSocket health monitoring
last_successful_emit: float = 0.0
emit_failures = 0
connection_issues_detected = 0


@socketio.on("connect")
def handle_connect():
    """Handle client connection"""
    print(f"[INFO] Client connected at {time.time()}")
    emit("connection_status", {"status": "connected", "timestamp": time.time()})


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection"""
    global connection_issues_detected
    connection_issues_detected += 1
    print(f"[WARNING] Client disconnected at {time.time()}")


@socketio.on("ping_heartbeat")
def handle_heartbeat():
    """Respond to client heartbeat to confirm connection is alive"""
    global last_successful_emit
    try:
        emit("pong_heartbeat", {"timestamp": time.time(), "server_health": "ok"})
        last_successful_emit = time.time()
    except Exception as e:
        print(f"[ERROR] Heartbeat failed: {e}")


@socketio.on("keypoints")
def handle_keypoints(payload: str) -> None:
    global frame_counter, frames_dropped, last_emit_time
    global classifications_completed, classification_times
    global last_successful_emit, emit_failures

    # Log frame arrival
    frame_counter += 1

    if stream_pose.stream_pose_client is None:
        emit("frame_result", {"error": "No model set"})
        return

    start_time = time.time()
    results = stream_pose.stream_pose_client.run_keypoint_pipeline(payload)
    current_time = time.time()
    speed = current_time - start_time

    # Calculate additional metrics
    time_since_last_classification = (
        current_time - stream_pose.stream_pose_client.last_prediction_timestamp
        if stream_pose.stream_pose_client.last_prediction_timestamp > 0
        else 0
    )

    # Calculate frames in current window
    frames_in_window = (
        len(stream_pose.stream_pose_client.frames)
        if hasattr(stream_pose.stream_pose_client, "frames")
        else 0
    )

    # Calculate current counter position
    (
        stream_pose.stream_pose_client.counter
        if hasattr(stream_pose.stream_pose_client, "counter")
        else 0
    )

    # Check if we're falling behind (pipeline taking longer than frame interval)

    # Emit the results back to the client
    if (
        results and stream_pose.stream_pose_client.current_classification is not None
    ):  # if we get some classification
        classification = stream_pose.stream_pose_client.current_classification
        predict_speed = stream_pose.stream_pose_client.prediction_processing_time
        classifications_completed += 1

        # Track classification times (keep last 10)
        classification_times.append(current_time)
        if len(classification_times) > 10:
            classification_times.pop(0)

        # Calculate comprehensive metrics
        update_freq = getattr(
            stream_pose.stream_pose_client, "update_frame_frequency", 25
        )
        frame_window_size = getattr(stream_pose.stream_pose_client, "frame_window", 30)
        frame_overlap = getattr(stream_pose.stream_pose_client, "frame_overlap", 5)

        # Time metrics
        time_since_emit = (
            current_time - last_emit_time if last_emit_time > 0 else float("inf")
        )
        burst_warning = time_since_emit < 0.5

        # Average time between classifications
        avg_time_between: float = 0.0
        if len(classification_times) > 1:
            intervals = [
                classification_times[i] - classification_times[i - 1]
                for i in range(1, len(classification_times))
            ]
            avg_time_between = sum(intervals) / len(intervals) if intervals else 0

        # Frame age calculations (30fps = 33.33ms per frame)
        frame_interval_ms = 1000 / 30  # Assume 30fps input
        oldest_frame_age_ms = frame_window_size * frame_interval_ms
        newest_frame_age_ms = frame_interval_ms

        # Processing modes and health
        can_process_every_frame = speed < frame_interval_ms / 1000
        max_sustainable_fps = 1.0 / speed
        input_fps = 30  # Could be dynamic later

        # Queue and processing status
        queue_depth = 0  # With our buffer limit, should stay at 0
        processing_mode = "sampling"  # Since we're not processing every frame
        if can_process_every_frame:
            processing_mode = "real_time"
        elif queue_depth > 0:
            processing_mode = "queuing"

        # System health assessment
        maintaining_rate = time_since_last_classification > 0
        using_latest_data = queue_depth == 0

        if maintaining_rate and using_latest_data and not burst_warning:
            system_health = "optimal"
        elif burst_warning or not maintaining_rate:
            system_health = "degraded"
        else:
            system_health = "good"

        # Capacity metrics
        min_interval = 0.8  # Our rate limiting interval
        # Use total processing time as the real bottleneck
        total_capacity_used = speed / min_interval if min_interval > 0 else 0
        inference_proportion = predict_speed / speed if speed > 0 else 0

        # Check for connection health issues
        time_since_last_emit = (
            current_time - last_successful_emit if last_successful_emit > 0 else 0
        )
        connection_health = "healthy"
        if time_since_last_emit > 5:  # No successful emit in 5 seconds
            connection_health = "degraded"
        if time_since_last_emit > 10:  # No successful emit in 10 seconds
            connection_health = "critical"

        return_payload = {
            # Core results
            "classification": classification,
            "timestamp": f"{time.time_ns()}",
            # Connection Health
            "connection_health": connection_health,
            "time_since_last_emit_ms": round(time_since_last_emit * 1000, 1),
            "emit_failures": emit_failures,
            "connection_issues": connection_issues_detected,
            # Classification Timing
            "classification_rate_hz": (
                round(1.0 / avg_time_between, 2) if avg_time_between > 0 else 0
            ),
            "time_since_last_classification_ms": round(
                time_since_last_classification * 1000, 1
            ),
            # Frame Window Details
            "frames_in_window": frames_in_window,
            "frames_per_classification": frame_window_size,  # Frames sent to model
            "frames_between_windows": update_freq,  # Frames between starts
            "frame_overlap": frame_overlap,
            # Window Performance Analysis
            "ideal_classification_interval_ms": round(
                update_freq * frame_interval_ms, 1
            ),  # Based on window step size
            "actual_classification_interval_ms": (
                round(avg_time_between * 1000, 1) if avg_time_between > 0 else 0
            ),
            # Processing Timeline Breakdown
            "frame_collection_time_ms": round(
                getattr(stream_pose.stream_pose_client, "frame_collection_time", 0)
                * 1000,
                1,
            ),
            "transformation_time_ms": round(
                getattr(stream_pose.stream_pose_client, "transformation_time", 0)
                * 1000,
                1,
            ),
            "mlflow_inference_time_ms": round(
                getattr(stream_pose.stream_pose_client, "mlflow_inference_time", 0)
                * 1000,
                1,
            ),
            # Total processing time already shown above in breakdown
            "total_processing_time_ms": round(speed * 1000, 1),
            # System State
            "processing_mode": processing_mode,
            "queue_depth": queue_depth,
            "frames_processed": frame_counter,
            "classifications_completed": classifications_completed,
            # Data Freshness
            "oldest_frame_age_ms": round(oldest_frame_age_ms, 1),
            "newest_frame_age_ms": round(newest_frame_age_ms, 1),
            "using_latest_data": using_latest_data,
            # Classification Capacity
            "can_process_every_frame": can_process_every_frame,
            "max_classifications_per_second": round(
                max_sustainable_fps, 1
            ),  # Based on total processing time
            "input_fps": input_fps,
            "total_capacity_used": round(
                total_capacity_used, 2
            ),  # Rate limit usage by total processing
            "inference_proportion": round(
                inference_proportion, 2
            ),  # Fraction of processing time that is inference vs overhead
            # Health Status
            "maintaining_classification_rate": maintaining_rate,
            "burst_warning": burst_warning,
            "system_health": system_health,
        }

        # Try to emit with error handling and tracking
        try:
            emit("frame_result", return_payload)
            last_emit_time = current_time
            last_successful_emit = current_time

            # Log if we recovered from issues
            if emit_failures > 0:
                print(f"[INFO] WebSocket recovered after {emit_failures} failures")
                emit_failures = 0

        except Exception as e:
            emit_failures += 1
            print(f"[ERROR] Failed to emit frame_result: {e}")
            print(f"[ERROR] Emit failures: {emit_failures}")

            # Try to force reconnection if too many failures
            if emit_failures > 10:
                print("[CRITICAL] Too many emit failures, attempting recovery")

                # Recovery attempt 1: Send a force_reconnect signal
                try:
                    emit(
                        "force_reconnect",
                        {"reason": "emit_failures", "count": emit_failures},
                    )
                    print("[INFO] Sent force_reconnect signal to client")
                except Exception:
                    pass

                # Recovery attempt 2: Try disconnect to force client reconnection
                if emit_failures > 20:
                    try:
                        print("[CRITICAL] Forcing disconnect for reconnection")
                        # Skip forced disconnection - let client handle recovery
                    except Exception:
                        pass

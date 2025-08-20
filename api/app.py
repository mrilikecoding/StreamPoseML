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

# Web Socket - TODO there is some optimization to be done here - need to look at options
Payload.max_decode_packets = 2000
socketio: SocketIO = SocketIO(
    app,
    async_mode="eventlet",
    ping_timeout=30,
    ping_interval=20,
    cors_allowed_origins="*",
)

# Tmp file upload - use absolute path to ensure we use the shared volume
UPLOAD_FOLDER = "/usr/src/app/tmp"
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
CORS(app,
     origins=whitelist,
     supports_credentials=True,
     expose_headers=["Content-Type", "Authorization"],
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"])
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
        extract_to = os.path.join(
            upload_folder, filename.rsplit(".", 1)[0]
        )

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


@socketio.on("keypoints")
def handle_keypoints(payload: str) -> None:
    if stream_pose.stream_pose_client is None:
        emit("frame_result", {"error": "No model set"})
        return

    start_time = time.time()
    results = stream_pose.stream_pose_client.run_keypoint_pipeline(payload)
    current_time = time.time()
    speed = current_time - start_time
    # Emit the results back to the client
    if (
        results and stream_pose.stream_pose_client.current_classification is not None
    ):  # if we get some classification
        classification = stream_pose.stream_pose_client.current_classification
        predict_speed = stream_pose.stream_pose_client.prediction_processing_time
        return_payload = {
            "classification": classification,
            "timestamp": f"{time.time_ns()}",
            "pipeline processing time (s)": speed,
            "prediction processing time (s)": predict_speed,
            "frame rate capacity (hz)": 1.0 / speed,
        }

        emit("frame_result", return_payload)

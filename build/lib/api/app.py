import time

from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from engineio.payload import Payload

from stream_pose_ml import stream_pose_client
from stream_pose_ml.blaze_pose import mediapipe_client
from stream_pose_ml.learning import trained_model
from stream_pose_ml.learning import sequence_transformer
from stream_pose_ml.learning import model_builder

from stream_pose_ml.actuators import bluetooth_device

### Set the model ###
# Load trained model into TrainedModel instance - note, models in local folder were saved
# via model builder, so use the retrieve_model_from_pickle method in the model_builder.
# Otherwise, any model with a "predict" method can be set on the trained_model instance.
mb = model_builder.ModelBuilder()
model_location = "./data/trained_models"
trained_model = trained_model.TrainedModel()


### Set the pose estimation client ###
mpc = mediapipe_client.MediaPipeClient(dummy_client=True)


### Create StreamPoseML client wrapper to instantiate from front end ###
class StreamPoseMLApp:
    def __init__(self):
        self.stream_pose_client = None

    def set_stream_pose_client(self, stream_pose_client):
        self.stream_pose_client = stream_pose_client

    def set_actuator(self, actuator="bluetooth_device"):
        if actuator == "bluetooth_device":
            self.actuator = bluetooth_device.BluetoothDevice()

    def actuate(self, data):
        self.actuator.send(data)
        return self.actuator.receive()


stream_pose = StreamPoseMLApp()

### Init Flask API ###
app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
app.debug = True

# TODO - make env dependent from config
whitelist = [
    "http://localhost:3000",
    "http://localhost:5001",
    "https://cdn.jsdelivr.net",
]
CORS(app, origins=whitelist)


@app.route("/")
def status():
    return "Server Ready"


### Application Routes ###
@app.route("/set_model", methods=["POST"])
def set_model():
    data = request.get_json()
    if "filename" not in data:
        return {"error": "No filename"}, 400

    trained_model_pickle_path = data["filename"]
    model, model_data = mb.retrieve_model_from_pickle(
        file_path=f"{model_location}/{trained_model_pickle_path}"
    )
    trained_model.set_model(model=model, model_data=model_data)

    ### Set the trained_models data transformer ###
    # TODO replace this with some kind of schema
    transformer = sequence_transformer.TenFrameFlatColumnAngleTransformer()
    trained_model.set_data_transformer(transformer)

    stream_pose.set_stream_pose_client(
        stream_pose_client.StreamPoseClient(
            mediapipe_client_instance=mpc,
            trained_model=trained_model,
            data_transformer=transformer,
            frame_window=10,
        )
    )

    # TODO - add a separate step for this and make configurable
    stream_pose.set_actuator()

    return {"result": f"Server Ready: classifier set to {trained_model_pickle_path}."}


### SocketIO Listeners ###

# Web Socket - TODO there is some optimization to be done here - need to look at these options
Payload.max_decode_packets = 500
socketio = SocketIO(
    app,
    async_mode="eventlet",
    ping_timeout=10,
    ping_interval=2,
    cors_allowed_origins="*",
)


@socketio.on("keypoints")
def handle_keypoints(payload: str) -> None:
    if stream_pose.stream_pose_client is None:
        emit("frame_result", {"error": "No model set"})
        return

    start_time = time.time()
    results = stream_pose.stream_pose_client.run_keypoint_pipeline(payload)
    speed = time.time() - start_time

    # Emit the results back to the client
    if (
        results and stream_pose.stream_pose_client.current_classification is not None
    ):  # if we get some classification
        classification = stream_pose.stream_pose_client.current_classification
        return_payload = {
            "classification": classification,
            "timestamp": f"{time.time_ns()}",
            "processing time (s)": speed,
            "frame rate capacity (hz)": 1.0 / speed,
        }

        emit("frame_result", return_payload)


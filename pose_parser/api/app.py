import time

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO, send, emit

from pose_parser import poser_client
from pose_parser.blaze_pose import mediapipe_client
from pose_parser.learning import trained_model
from pose_parser.learning import sequence_transformer
from pose_parser.learning import model_builder

### Set the model ###
# Load trained model into TrainedModel instance - note, models in local folder were saved
# via model builder, so use the retrieve_model_from_pickle method in the model_builder.
# Otherwise, any model with a "predict" method can be set on the trained_model instance.
mb = model_builder.ModelBuilder()
# TODO grab this path from config
trained_model_pickle_path = (
    "./data/trained_models/gradient-boost-1683824097745362000.pickle"
)
trained_model = trained_model.TrainedModel()
trained_model.set_model(
    mb.retrieve_model_from_pickle(file_path=trained_model_pickle_path),
    notes="Gradient Boost trained on 10 frame window, flat columns, all angles + distances",
)

### Set the trained_models data transformer ###
transformer = sequence_transformer.TenFrameFlatColumnAngleTransformer()
trained_model.set_data_transformer(transformer)

### Set the pose estimation client ###
mpc = mediapipe_client.MediaPipeClient(dummy_client=True)

### Init the Poser Client ###
pc = poser_client.PoserClient(
    mediapipe_client_instance=mpc,
    trained_model=trained_model,
    data_transformer=transformer,
    frame_window=10,
)

### Init Flask API ###
app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
app.debug = True

# TODO - make env dependent from config
whitelist = [
    "http://localhost:3000",
    "http://localhost:5000",
]
CORS(app, origins=whitelist)

# Web Socket
socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins="*")


@socketio.on("frame")
def handle_frame(payload: str) -> None:
    """
    Handle incoming video frames from the client.

    This event handler is triggered when a 'frame' event is received from the client side.
    It processes the frame data (e.g., perform keypoint extraction) and sends the results back to the client.

    Args:
        payload: str
            The payload of the 'frame' event, containing the base64 encoded frame data.
    """
    image = pc.convert_base64_to_image_array(payload)
    # image = pc.preprocess_image(image)

    start_time = time.time()
    results = pc.run_frame_pipeline(image)
    speed = time.time() - start_time

    # Emit the results back to the client
    if (
        results and pc.current_classification is not None
    ):  # if we get some classification
        return_payload = {
            "classification": pc.current_classification,
            "timestamp": f"{time.time_ns()}",
            "processing time (s)": speed,
            "frame rate capacity (hz)": 1.0 / speed,
        }
        emit("frame_result", return_payload)

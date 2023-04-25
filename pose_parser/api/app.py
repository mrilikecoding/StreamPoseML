import time

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO, send, emit

from pose_parser import poser_client
from pose_parser.blaze_pose import mediapipe_client
from pose_parser.learning import trained_model
from pose_parser.learning import sequence_transformer

mpc = mediapipe_client.MediaPipeClient(dummy_client=True)

# TODO
# 1) Load trained model into TrainedModel instance
# model = trained_model.Trained_Model()
# model = model.load_trained_model(path/to/model)
# transformer = SomeConcreteSequenceTransformerCorrespondingToModel()
# model.set_data_transformer(transformer)
# pc = poser_client.PoserClient(mediapipe_client_instance=mpc, trained_model=model)
pc = poser_client.PoserClient(mediapipe_client_instance=mpc)


# 2) Figure out data transformation layer
#    - this is I think writing a separate transformer class for different kidns of models
#    - pass into TrainedModel - then the data is transformed before #predict
# 3) call predict

# Flask
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

    # Do stuff - send to mediapipe client etc...
    start_time = time.time()
    results = pc.run_frame_pipeline(image)
    speed = time.time() - start_time

    # Emit the results back to the client
    if True:  # if we get some classification
        return_payload = {
            "timestamp": f"{time.time_ns()}",
            "processing time (s)": speed,
            "frame rate capacity (hz)": 1.0 / speed,
        }
        emit("frame_result", return_payload)

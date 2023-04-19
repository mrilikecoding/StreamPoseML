import time

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO, send, emit

from pose_parser import poser_client
from pose_parser.blaze_pose import mediapipe_client

mpc = mediapipe_client.MediaPipeClient(dummy_client=True)
pc = poser_client.PoserClient(mediapipe_client_instance=mpc)


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
            "timestamp": f"SUCCESS {time.time_ns()}",
            "processing time (s)": speed,
            "frame rate capacity (hz)": 1.0 / speed,
        }
        emit("frame_result", return_payload)

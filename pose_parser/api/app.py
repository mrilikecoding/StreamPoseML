import base64
import time

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO, send, emit


app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'

# Enable CORS for all routes and origins
# TODO - make env dependent from config
whitelist = [
    "http://localhost:3000",
    "http://localhost:5000",
]
CORS(app, origins=whitelist)
socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins="*")


@app.route("/")
def hello_world():
    return "Hello World"


@socketio.on("frame")
def handle_frame(payload: str) -> None:
    """
    Handle incoming video frames from the client.

    This event handler is triggered when a 'frame' event is received from the client side.
    It processes the frame data (e.g., perform keypoint extraction) and sends the results back to the client.

    Args:
        payload: The payload of the 'frame' event, containing the base64 encoded frame data.
    """
    # Decode the base64 encoded frame data to an image
    img_data = base64.b64decode(payload.split(",")[1])

    # Do stuff - send to mediapipe client etc...

    # Emit the results back to the client
    if True:  # if we get some classification
        results = {"TEST": f"SUCCESS {time.time_ns()}"}
        emit("frame_result", results)


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)

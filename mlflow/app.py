import logging
import os
import signal
import subprocess
import threading
import time

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Global variable to keep track of the model server process
model_server_process = None


@app.route("/status", methods=["GET"])
def status():
    return (
        jsonify(
            {"status": "running", "message": "MLflow model server is up and running."}
        ),
        200,
    )


@app.route("/ping", methods=["POST"])
def ping():
    if model_server_process:
        return (
            jsonify({"status": "success", "message": "Model server is running."}),
            200,
        )
    else:
        return jsonify({"status": "error", "message": "Model server not running."}), 500


@app.route("/load_model", methods=["POST"])
def load_model():
    global model_server_process

    # Get the model path from the request
    data = request.get_json()
    model_path = data.get("model_path")

    if not model_path or not os.path.exists(model_path):
        print("invalid model path")
        return jsonify({"status": "error", "message": "Invalid model path."}), 400

    # Terminate existing model server if running
    if model_server_process:
        os.killpg(os.getpgid(model_server_process.pid), signal.SIGTERM)
        model_server_process = None
        time.sleep(2)  # Wait for process to terminate

    # Start the MLflow model serving process
    command = [
        "mlflow",
        "models",
        "serve",
        "-m",
        model_path,
        "-p",
        "1234",  # Use a different port
        "-h",
        "0.0.0.0",
    ]

    model_server_process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,  # Start the process in a new process group
    )

    # Start threads to read subprocess output
    stdout_thread = threading.Thread(
        target=read_subprocess_output, args=(model_server_process.stdout, logger.info)
    )
    stderr_thread = threading.Thread(
        target=read_subprocess_output, args=(model_server_process.stderr, logger.error)
    )
    stdout_thread.start()
    stderr_thread.start()

    # Optionally, start a monitoring thread
    monitor_thread = threading.Thread(target=monitor_model_server)
    monitor_thread.start()

    return jsonify({"status": "success", "message": "Model server started."}), 200


def read_subprocess_output(pipe, logger_method):
    for line in iter(pipe.readline, b""):
        decoded_line = line.decode().rstrip()
        logger_method(decoded_line)


def monitor_model_server():
    global model_server_process
    while True:
        if model_server_process:
            return_code = model_server_process.poll()
            if return_code is not None:
                logger.error(f"Model server process exited with code {return_code}")
                # Optionally restart the process or notify
                break
        time.sleep(5)


@app.route("/invocations", methods=["POST"])
def invocations():
    if not model_server_process:
        return jsonify({"status": "error", "message": "Model server not running."}), 500

    # Forward the request to the MLflow model server
    try:
        response = requests.post(
            "http://localhost:1234/invocations",
            data=request.data,
            headers={"Content-Type": request.headers.get("Content-Type")},
        )
        res = (response.content, response.status_code, response.headers.items())
        print(res)
        return res
    except requests.exceptions.ConnectionError:
        return (
            jsonify(
                {"status": "error", "message": "Failed to connect to model server."}
            ),
            500,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)

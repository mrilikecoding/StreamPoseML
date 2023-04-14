""" Here is the beginning of an API layer... """
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS for all routes and origins
# TODO - restrict only to react app
CORS(app)


@app.route("/")
def hello_world():
    return "Hello World"


@app.route("/process_videos")
def process_videos():
    return "Process Videos!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

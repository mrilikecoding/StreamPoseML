"""
Fixtures for API tests.
"""

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask
from flask_socketio import SocketIO

# Project root is already in sys.path via the pytest.ini or setup.py configuration


# Mark API tests
def pytest_configure(config):
    """Add api_test marker."""
    config.addinivalue_line("markers", "api_test: mark test as an API test")


@pytest.fixture(scope="function")
def simple_test_app():
    """Create a simple Flask app for testing without dependencies."""
    app = Flask(__name__)
    app.config["TESTING"] = True
    socketio = SocketIO(app)

    return app, socketio


@pytest.fixture(scope="function")
def mock_api_dependencies():
    """
    Fixture that patches the API dependencies only for that specific test.
    """
    # Set up all your mocks here
    mocks = {}

    # Create API module level mocks
    bluetooth_device_mock = MagicMock()
    stream_pose_client_mock = MagicMock()
    ml_flow_client_mock = MagicMock()
    mediapipe_client_mock = MagicMock()
    trained_model_mock = MagicMock()
    sequence_transformer_mock = MagicMock()
    model_builder_mock = MagicMock()

    mediapipe_client_mock.MediaPipeClient.return_value.serialize_pose_landmarks.return_value = [  # noqa: E501
        {"x": 0.1, "y": 0.2, "z": 0.3} for _ in range(33)
    ]

    transformer = MagicMock()
    transformer.transform.return_value = ({"data": "transformed"}, {"meta": "data"})
    sequence_transformer_mock.TenFrameFlatColumnAngleTransformer.return_value = (
        transformer
    )
    sequence_transformer_mock.MLFlowTransformer.return_value = transformer

    trained_model_mock.TrainedModel.return_value.model_data = {"X_test": MagicMock()}
    trained_model_mock.TrainedModel.return_value.model_data["X_test"].columns = [
        "col1",
        "col2",
    ]
    trained_model_mock.TrainedModel.return_value.predict.return_value = [1]

    # Create modules to mock with sys.modules
    modules_to_mock = {
        "stream_pose_ml": MagicMock(),
        "stream_pose_ml.blaze_pose": MagicMock(),
        "stream_pose_ml.blaze_pose.mediapipe_client": mediapipe_client_mock,
        "stream_pose_ml.learning": MagicMock(),
        "stream_pose_ml.learning.trained_model": trained_model_mock,
        "stream_pose_ml.transformers": MagicMock(),
        "stream_pose_ml.transformers.sequence_transformer": sequence_transformer_mock,
        "stream_pose_ml.learning.model_builder": model_builder_mock,
        "stream_pose_ml.stream_pose_client": stream_pose_client_mock,
        "stream_pose_ml.ml_flow_client": ml_flow_client_mock,
    }

    # Setup module relationships
    modules_to_mock["stream_pose_ml"].actuators = modules_to_mock[
        "stream_pose_ml.actuators"
    ]
    modules_to_mock["stream_pose_ml"].blaze_pose = modules_to_mock[
        "stream_pose_ml.blaze_pose"
    ]
    modules_to_mock["stream_pose_ml"].learning = modules_to_mock[
        "stream_pose_ml.learning"
    ]
    modules_to_mock["stream_pose_ml"].transformers = modules_to_mock[
        "stream_pose_ml.transformers"
    ]
    modules_to_mock["stream_pose_ml"].stream_pose_client = stream_pose_client_mock
    modules_to_mock["stream_pose_ml"].ml_flow_client = ml_flow_client_mock

    # Apply patch to sys.modules
    patcher = patch.dict("sys.modules", modules_to_mock)
    patcher.start()

    # Also define paths to mock directly in app.py
    patches = [
        patch("api.app.bluetooth_device", bluetooth_device_mock),
        patch("api.app.stream_pose_client", stream_pose_client_mock),
        patch("api.app.ml_flow_client", ml_flow_client_mock),
        patch("api.app.mediapipe_client", mediapipe_client_mock),
        patch("api.app.trained_model", trained_model_mock),
        patch("api.app.sequence_transformer", sequence_transformer_mock),
        patch("api.app.model_builder", model_builder_mock),
    ]

    # Start all patches
    for p in patches:
        p.start()
        mocks[p.target] = p

    yield mocks

    # Stop all patches
    for p in patches:
        p.stop()
    patcher.stop()

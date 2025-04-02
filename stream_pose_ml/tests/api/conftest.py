"""
Fixtures for API tests.
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import inspect

# Add paths
parent_dir = Path(__file__).parent.parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Mark API tests
def pytest_configure(config):
    """Add api_test marker."""
    config.addinivalue_line("markers", "api_test: mark test as an API test")

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
    
    # Set up mock return values and behaviors
    bluetooth_device_mock.BluetoothDevice.return_value.receive.return_value = "mock_response"
    mediapipe_client_mock.MediaPipeClient.return_value.serialize_pose_landmarks.return_value = [
        {"x": 0.1, "y": 0.2, "z": 0.3} for _ in range(33)
    ]
    
    transformer = MagicMock()
    transformer.transform.return_value = ({"data": "transformed"}, {"meta": "data"})
    sequence_transformer_mock.TenFrameFlatColumnAngleTransformer.return_value = transformer
    sequence_transformer_mock.MLFlowTransformer.return_value = transformer
    
    trained_model_mock.TrainedModel.return_value.model_data = {
        "X_test": MagicMock() 
    }
    trained_model_mock.TrainedModel.return_value.model_data["X_test"].columns = ["col1", "col2"]
    trained_model_mock.TrainedModel.return_value.predict.return_value = [1]
    
    # Define paths to mock
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
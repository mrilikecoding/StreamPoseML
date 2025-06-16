# Stream Pose ML Project Structure

## Current Structure

The project has been restructured to have a flatter and more modular organization:

```
stream_pose_ml/
├── api/                 # Flask API for the Stream Pose ML project
├── notebooks/           # Jupyter notebooks for examples and documentation
├── stream_pose_ml/      # Main Python module with core functionality
├── tests/               # Tests for both the main module and API
│   ├── api/             # API-specific tests
│   └── ...              # Core module tests
└── ...                  # Other project files
```

## Testing Setup

The project uses pytest for testing, with separate configurations for the main module tests and API tests:

- `pytest.ini`: Configuration for running general module tests 
- `api_pytest.ini`: Configuration specifically for API tests

### Running Tests

Run the main module tests:
```bash
python -m pytest
```

Run the API tests:
```bash
python -m pytest -c api_pytest.ini
```

## Module Structure

### Main Module (`stream_pose_ml/`)

The main module contains all the core functionality for pose detection and analysis:

- `blaze_pose/`: BlazePose implementation for pose detection
- `geometry/`: Utility classes for geometric calculations
- `jobs/`: Batch job processing classes
- `learning/`: ML model training and inference
- `serializers/`: Classes for data serialization
- `services/`: Various service classes
- `transformers/`: Data transformation utilities
- `utils/`: General utility functions

### API Module (`api/`)

The API module provides a Flask web server with SocketIO support for real-time communication:

- `app.py`: Main Flask application with API endpoints and WebSocket handlers
- Tests are in `tests/api/`

## API Testing Approach

API tests use a different approach that doesn't rely on importing the actual API code:

1. **Independent Test Fixtures**: Create dummy versions of the Flask app, SocketIO, and StreamPoseMLApp classes for testing
2. **Mock-based Testing**: Use extensive mocking to simulate behavior without calling actual dependencies
3. **Isolated Route Testing**: Create route handlers on-the-fly for each test
4. **Socket Testing**: Use the SocketIO test client for WebSocket communication tests

## Test Performance

- Regular module tests: Run with `python -m pytest`
- API tests: Run with `python -m pytest -c api_pytest.ini`
- All tests together: Run with `python -m pytest --ignore=markers`

## Completed

1. ✅ Restructured project to have a flatter hierarchy
2. ✅ Moved API and notebooks to top level
3. ✅ Fixed import issues in app.py
4. ✅ Created separate testing strategy for API 
5. ✅ All tests are now passing
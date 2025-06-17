# API Tests

This directory contains tests for the Stream Pose ML API.

## Testing Approach

The API tests are designed to be run independently from the core module tests. This separation is necessary because the API depends on the core module, but we want to test the API's functionality without getting entangled in the internals of the core module.

The tests use the following approaches:

1. **Independent Test Fixtures**: We create dummy versions of the Flask app, SocketIO, and StreamPoseMLApp classes for testing, rather than importing the real ones from the API. This prevents import issues and makes tests more isolated.

2. **Mock-based Testing**: We use extensive mocking to simulate the behavior of dependencies without actually calling them.

3. **Isolated Route Testing**: We create route handlers on-the-fly for each test, which allows us to precisely control the behavior of each endpoint.

## Running Tests

To run the API tests independently:

```bash
python -m pytest -c api_pytest.ini
```

This uses a dedicated pytest configuration that focuses only on the API tests.

## Test Structure

- `test_app.py`: Contains tests for the Flask app's routes and helper functions
- `test_integration.py`: Contains tests for integration points between the API and client components

## Notes on Implementation

- The `simple_test_app` and `dummy_stream_pose_ml_app` fixtures provide clean testing environments for each test.
- API tests don't import directly from the API except when necessary, to avoid dependencies.
- The SocketIO tests use real SocketIO test clients rather than mocking the socket behavior.
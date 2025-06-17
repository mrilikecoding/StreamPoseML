# Stream Pose ML API

This directory contains the Flask API for the Stream Pose ML project.

## Running the API

```bash
python -m api.app
```

## Running API Tests

The API tests are configured to run separately from the main module tests to avoid import conflicts. To run the API tests:

```bash
# From the root directory
pytest -c api_pytest.ini
```

This will use the specialized pytest configuration in `api_pytest.ini` that targets only the API tests.
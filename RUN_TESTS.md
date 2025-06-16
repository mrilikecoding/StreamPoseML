# How to Run Tests

After restructuring the project, use these commands to run tests:

## Option 1: Run tests using our test runner

```bash
python run_all_tests.py
```

This will run both API tests (in /tests) and module tests (in /stream_pose_ml/tests).

## Option 2: Run tests directly with pytest

Make sure you install the package in development mode first:
```bash
pip install -e .
```

Then run all tests:
```bash
python -m pytest
```

Or run specific test directories:
```bash
# API tests only
python -m pytest tests/

# Module tests only
python -m pytest stream_pose_ml/tests/
```

## If you get import errors:

If you're getting import errors, the most reliable approach is:

```bash
# Install the package in development mode
pip install -e .

# Fix the module imports (one-time)
./fix_imports.sh

# Run the tests with our custom script
python run_all_tests.py
```

This ensures the Python path is set correctly and all imports can be resolved.

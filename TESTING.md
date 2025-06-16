# Testing Guide for stream_pose_ml

After restructuring the project to move the API and notebooks to the top level, this guide explains how to run tests for both the API and module components.

## Test Structure

The project has two main test directories:

1. **API Tests**: Located in `/tests`
   - Tests for the API functionality
   - Uses mocks for stream_pose_ml module components

2. **Module Tests**: Located in `/stream_pose_ml/tests`
   - Tests for the core stream_pose_ml module functionality
   - Includes tests for geometry, blaze_pose, and other components

## Running Tests

### Option 1: Use the provided test runner script (Recommended)

We've created a convenience script that handles dependencies and runs all tests in the correct order:

```bash
# First make the script executable
chmod +x run_all_tests.py

# Run all tests
./run_all_tests.py
```

This will:
1. Install necessary dependencies
2. Run the API tests first
3. Run the module tests second
4. Report overall success or failure

### Option 2: Run tests manually with pytest

You can also run tests directly with pytest:

```bash
# Install the package in development mode
pip install -e .

# Install test dependencies
pip install pytest numpy

# Run all tests
python -m pytest

# Run only API tests
python -m pytest tests/

# Run only module tests
python -m pytest stream_pose_ml/tests/

# Run a specific test file
python -m pytest stream_pose_ml/tests/geometry/test_distance.py
```

## Understanding Import Paths

After restructuring, we've ensured tests can find modules by:

1. Adding the project root to `sys.path` in test files
2. Configuring pytest to look in both test directories
3. Using absolute imports for consistency

## Troubleshooting

If you encounter import errors:

1. **Check if the package is installed in development mode**:
   ```bash
   pip install -e .
   ```

2. **Verify Python path**:
   ```python
   import sys
   print(sys.path)
   ```
   The project root should be in the path.

3. **Check import statements**:
   Imports should be absolute, e.g., `from stream_pose_ml.geometry.distance import Distance`

4. **Try running with the helper script**:
   ```bash
   ./run_all_tests.py
   ```

## Code Structure After Restructuring

The new structure is:

```
/stream_pose_ml         # Project root
├── api/                # API module (moved from stream_pose_ml/stream_pose_ml/api)
├── notebooks/          # Notebooks (moved from stream_pose_ml/stream_pose_ml/notebooks)
├── stream_pose_ml/     # Core module (flattened)
│   ├── blaze_pose/     # Blaze pose components
│   ├── geometry/       # Geometry components
│   ├── ...             # Other submodules
│   └── tests/          # Module tests
└── tests/              # API tests
```

This structure allows for cleaner imports and better organization while maintaining backward compatibility with existing code.
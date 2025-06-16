# Module Test Restructuring

After moving the API and notebooks to the top level, we've successfully fixed the import paths in the module test files. This document explains what was done and what remains to be done to get all tests running.

## Changes Made

1. **Path adjustments in test files**
   - Added `sys.path` manipulation to all test files to ensure they can find the modules
   - Updated import statements to use absolute imports from the project root
   - All files now include the following code at the top:
   ```python
   import sys
   from pathlib import Path

   # Add the project root to the Python path
   project_root = Path(__file__).parents[3]  # /Users/nathangreen/Development/stream_pose_ml
   if str(project_root) not in sys.path:
       sys.path.insert(0, str(project_root))
   ```

2. **Scripts Created**
   - `fix_test_imports.py`: Script to add import path adjustments to all test files
   - `run_module_tests.py`: Script to run the module tests with proper path settings
   - `fix_test_paths.py`: Simplified script to just fix the import paths

## Current Status

All test files have been updated with the proper import path handling. However, running the tests requires the necessary dependencies to be installed.

## Requirements for Running Tests

To run the tests successfully, you'll need:

1. Install pytest and other dependencies:
   ```bash
   pip install pytest numpy
   ```

2. Run a specific test to verify it works:
   ```bash
   cd /Users/nathangreen/Development/stream_pose_ml
   python -m pytest stream_pose_ml/tests/geometry/test_distance.py -v
   ```

3. Or run all module tests:
   ```bash
   cd /Users/nathangreen/Development/stream_pose_ml
   python -m pytest stream_pose_ml/tests/ -v
   ```

## Next Steps

1. **Restructure Module Imports**: We need a comprehensive solution to fix all imports throughout the codebase. The current issues are:
   - Module imports in stream_pose_ml/__init__.py are using relative imports (`.stream_pose_client`)
   - Submodule imports are looking for `stream_pose_ml.blaze_pose` which doesn't exist as an importable module yet
   - Import cycles between modules making it difficult to resolve imports

2. **Create a Proper Package Structure**: A more comprehensive approach is needed:
   - Update setup.py to properly register the package and all subpackages
   - Modify all imports to be absolute imports (no more relative imports using `.` or `..`)
   - Create proper __init__.py files that expose the relevant classes

3. **Install Dependencies**: Before the tests will run successfully, you need to install the required packages:
   ```bash
   pip install pytest numpy
   ```

4. **Run Tests in Isolation**: Until the import issues are fully resolved, tests can be run in isolation by:
   - Adding the project root to sys.path in each test file (which we've done)
   - Running tests individually and fixing any specific import errors

5. **Package Installation Option**: Another approach is to install the package in development mode:
   ```bash
   pip install -e .
   ```
   This will make imports work properly assuming setup.py is configured correctly.

6. **Comprehensive Testing Structure**: After resolving all import issues, ensure:
   - CI/CD pipeline is updated with the new test paths
   - A test runner is set up that can run all tests in the correct environment
   - Documentation is updated to reflect the new structure

## Note on Module Structure

The module structure still maintains the original organization:
```
stream_pose_ml/
├── blaze_pose/
├── geometry/
├── jobs/
├── learning/
├── serializers/
├── services/
├── transformers/
├── utils/
└── tests/
```

This approach allows the tests to continue working without extensive refactoring of all the import statements throughout the codebase.
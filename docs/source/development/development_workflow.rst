Development Workflow
====================

Getting Started with StreamPoseML Development
-------------------------------------------

This guide will help you set up your development environment and understand the workflow for contributing to StreamPoseML.

Development Setup
---------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/mrilikecoding/StreamPoseML.git
      cd StreamPoseML

2. Install in development mode:

   .. code-block:: bash

      uv sync --extra dev

   This will install StreamPoseML along with all development dependencies.

3. Check that everything is working:

   .. code-block:: bash

      make test

Project Structure
---------------

The StreamPoseML project is organized as follows:

* ``stream_pose_ml/`` - The main Python package
  * ``blaze_pose/`` - Pose detection and keypoint extraction
  * ``geometry/`` - Geometric calculations (angles, distances, etc.)
  * ``jobs/`` - High-level workflow jobs
  * ``learning/`` - Machine learning components
  * ``serializers/`` - Data serialization utilities
  * ``services/`` - Core processing services
  * ``transformers/`` - Data transformation utilities
  * ``utils/`` - Helper functions
* ``api/`` - Flask API for model serving
* ``docs/`` - Documentation source files
* ``web_ui/`` - React frontend application
* ``mlflow/`` - MLflow integration for model serving
* ``example_data/`` - Sample data for testing and examples

Makefile Commands
---------------

StreamPoseML includes a comprehensive Makefile to help with common development tasks:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Command
     - Description
   * - ``make build_images``
     - Build and push Docker images
   * - ``make start``
     - Start the application using pre-built DockerHub images
   * - ``make start-debug``
     - Start the application with pre-built images and debug output
   * - ``make start-dev``
     - Start the application by building from local source code (development mode)
   * - ``make stop``
     - Stop the application containers
   * - ``make test``
     - Run all tests
   * - ``make test-core``
     - Run tests for the stream_pose_ml package
   * - ``make test-api``
     - Run tests for the API
   * - ``make lint``
     - Format Python code using Black
   * - ``make lint-check``
     - Check Python code formatting with Black (without modifying)
   * - ``make docs``
     - Build Sphinx documentation
   * - ``make docs-versioned``
     - Build versioned Sphinx documentation
   * - ``make docs-clean``
     - Clean documentation build directory
   * - ``make clean``
     - Clean up temporary Docker resources
   * - ``make help``
     - Show the help message with all available commands

Development Workflow
------------------

1. **Create a feature branch**:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. **Make changes and run tests**:

   As you make changes, regularly run the tests to ensure everything is working:

   .. code-block:: bash

      make test-core  # Test just the Python package
      make lint       # Format your code

3. **Update documentation**:

   Update any relevant documentation and build it locally to check:

   .. code-block:: bash

      make docs
      # Documentation will be available in docs/build/html

4. **Run the application locally**:

   Test your changes with the full application:

   .. code-block:: bash

      make start-dev

5. **Commit your changes**:

   Follow the conventional commits standard for commit messages:

   .. code-block:: bash

      git commit -m "feat: add new feature X"
      # Common prefixes: feat, fix, docs, style, refactor, test, chore

6. **Submit a pull request**:

   Push your changes and create a pull request on GitHub for review.

Docker Development
----------------

For working with the Docker-based components:

1. **Build local images**:

   .. code-block:: bash

      make build_images

2. **Run with local changes**:

   .. code-block:: bash

      make start-dev

3. **Inspect running containers**:

   .. code-block:: bash

      docker ps
      docker logs streamposeml_api_1  # Replace with container name from docker ps

Documentation Development
----------------------

The documentation is built with Sphinx:

1. **Install documentation dependencies**:

   .. code-block:: bash

      pip install -e .[docs]

2. **Build documentation**:

   .. code-block:: bash

      make docs

3. **View documentation**:

   Open `docs/build/html/index.html` in your browser.

4. **Add new pages**:

   Create new `.rst` files in the appropriate directory under `docs/source/` and add them to the relevant toctree in the parent directory's index.rst.

Handling Model Files
------------------

When working with trained models:

1. **Keep small example models in Git**:
   - Small trained models used for tests and examples can be committed to the repository under `example_data/trained_models/`

2. **Use Git LFS for larger models**:
   - For larger model files, use Git Large File Storage
   - Initialize LFS: `git lfs install`
   - Track model files: `git lfs track "*.pkl" "*.h5"`

3. **Document models clearly**:
   - Include a README in each model directory explaining what the model does, how it was trained, and example usage

Getting Help
----------

If you need assistance with development:

- Check the existing documentation
- Look through the examples in `notebooks/`
- File an issue on GitHub
- Reach out to the maintainers via GitHub Discussions

Happy coding!
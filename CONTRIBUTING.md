# How to Contribute

Your contributions would be welcome! How can this project benefit more people?

## Getting Started

To get your environment set up to run StreamPoseML, you'll need Python and Pip installed on your machine. Ideally you'll want an isolated Python environment. If you want to run the web application, a Docker environment is recommended.

### Installation for Development

```bash
# Install in development mode with dev dependencies
pip install -e .[dev]
```

## Workflow

Follow the [GitHub Flow Workflow](https://guides.github.com/introduction/flow/):

1.  Fork the project
1.  Check out the `main` branch
1.  Create a feature branch
1.  Write code and tests for your change
1.  From your branch, make a pull request against `https://github.com/mrilikecoding/StreamPoseML`
1.  Get your change reviewed
1.  Wait for your change to be pulled into `https://github.com/mrilikecoding/StreamPoseML/main`
1.  Delete your feature branch

## Development Workflow

### Make Targets

The project includes several make targets to streamline common development tasks:

```bash
# Show all available commands
make help
```

#### Running the Application

```bash
# Start using pre-built DockerHub images
make start

# Start with debug output
make start-debug

# Start with local code (development mode)
make start-dev

# Stop the application
make stop
```

#### Testing

```bash
# Run all tests
make test

# Run only stream_pose_ml package tests
make test-core

# Run only API tests
make test-api
```

Tests use `pytest`. Run `python setup.py` to set paths correctly before running tests manually.

## Style

Use `black` with defaults for Python code.

```bash
# Format Python code with Black
make lint

# Check Python code formatting with Black (without modifying)
make lint-check
```

## Issues

When creating an issue please try to ahere to the following format:

    module-name: One line summary of the issue (less than 72 characters)

    ### Expected behavior

    As concisely as possible, describe the expected behavior.

    ### Actual behavior

    As concisely as possible, describe the observed behavior.

    ### Steps to reproduce the behavior

    List all relevant steps to reproduce the observed behavior.

## Documentation

We also welcome improvements to the project tests or project docs. Please file an [issue](https://github.com/mrilikecoding/StreamPoseML/issues/new).

### Building Documentation

```bash
# Build Sphinx documentation
make docs

# Build versioned Sphinx documentation
make docs-versioned

# Clean documentation build directory
make docs-clean
```

Documentation is built using Sphinx and will be available in the `docs/build/html` directory after building.

## First Contributions

If you are a first time contributor, familiarize yourself with the:
* [GitHub Flow Workflow](https://guides.github.com/introduction/flow/)

## Docker Image Management

```bash
# Build and push Docker images
make build_images

# Clean up temporary Docker resources
make clean
```

# License

By contributing your code, you agree to license your contribution under the
terms of the [LICENSE](https://github.com/mrilikecoding/StreamPoseML/blob/main/LICENSE).

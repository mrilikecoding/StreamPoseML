# Makefile for StreamPoseML

# Define paths to scripts
BUILD_SCRIPT = bin/build_images.sh
START_SCRIPT = bin/start_app.sh
STOP_SCRIPT = bin/stop_app.sh

# Default target
.PHONY: all
all: build_images

# Docker image build target
.PHONY: build_images
build_images:
	@echo "Running the Docker image build and push process..."
	@bash $(BUILD_SCRIPT)

# Start application
.PHONY: start
start:
	@echo "Starting StreamPoseML application..."
	@bash $(START_SCRIPT)

# Start application with debug mode
.PHONY: start-debug
start-debug:
	@echo "Starting StreamPoseML application in debug mode..."
	@bash $(START_SCRIPT) --debug

# Start application with local code (development mode)
.PHONY: start-dev
start-dev:
	@echo "Starting StreamPoseML application with local code (development mode)..."
	docker compose -f docker-compose.local.yml up --build

# Stop application
.PHONY: stop
stop:
	@echo "Stopping StreamPoseML application..."
	@bash $(STOP_SCRIPT)

# Test targets
.PHONY: test
test: test-core test-api
	@echo "All tests completed."

# Test the core stream_pose_ml package
.PHONY: test-core
test-core:
	@echo "Running stream_pose_ml package tests..."
	@uv run pytest stream_pose_ml/tests

# Test the API
.PHONY: test-api
test-api:
	@echo "Running API tests..."
	@uv run pytest api/tests -c api_pytest.ini

# Format Python code using Black
.PHONY: lint
lint:
	@echo "Formatting Python code with Black..."
	@uv run black stream_pose_ml api mlflow
	
# Check Python code formatting with Black (without modifying)
.PHONY: lint-check
lint-check:
	@echo "Checking Python code formatting with Black..."
	@uv run black --check stream_pose_ml api mlflow

# Clean target
.PHONY: clean
clean:
	@echo "Cleaning up temporary Docker resources..."
	@docker system prune -f

# Documentation targets
.PHONY: docs docs-versioned docs-clean

# Build the documentation
docs:
	@echo "Building Sphinx documentation..."
	@uv add --dev sphinx sphinx-multiversion sphinx-autodoc-typehints tomli
	@cd docs && sphinx-build -b html source build/html
	@echo "Documentation built in docs/build/html"

# Build versioned documentation
docs-versioned:
	@echo "Building versioned Sphinx documentation..."
	@uv add --dev sphinx sphinx-multiversion sphinx-autodoc-typehints tomli
	@cd docs && sphinx-multiversion source build/html
	@echo "Versioned documentation built in docs/build/html"

# Clean the documentation build directory
docs-clean:
	@echo "Cleaning documentation build directory..."
	@rm -rf docs/build

# Help target
.PHONY: help
help:
	@echo "StreamPoseML Makefile targets:"
	@echo "  all          - Run the default target (build_images)"
	@echo "  build_images - Build and push Docker images"
	@echo "  start        - Start the application using pre-built DockerHub images"
	@echo "  start-debug  - Start the application with pre-built images and debug output"
	@echo "  start-dev    - Start the application by building from local source code (development mode)"
	@echo "  stop         - Stop the application containers"
	@echo "  test         - Run all tests"
	@echo "  test-core    - Run tests for the stream_pose_ml package"
	@echo "  test-api     - Run tests for the API"
	@echo "  lint         - Format Python code using Black"
	@echo "  lint-check   - Check Python code formatting with Black (without modifying)"
	@echo "  docs         - Build Sphinx documentation"
	@echo "  docs-versioned - Build versioned Sphinx documentation"
	@echo "  docs-clean   - Clean documentation build directory"
	@echo "  clean        - Clean up temporary Docker resources"
	@echo "  help         - Show this help message"

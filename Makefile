# Makefile for building and pushing Docker images

# Define the path to the build script
BUILD_SCRIPT = bin/build_images.sh

# Default target
.PHONY: all
all: build_images

# Target to run the build script
.PHONY: build_images
build_images:
	@echo "Running the Docker image build and push process..."
	@bash $(BUILD_SCRIPT)

# Clean target (optional, if you want to add cleanup functionality)
.PHONY: clean
clean:
	@echo "Cleaning up temporary Docker resources..."
	@docker system prune -f

# Help target
.PHONY: help
help:
	@echo "Makefile targets:"
	@echo "  all          - Run the default target (build_images)"
	@echo "  build_images - Build and push Docker images using the build script"
	@echo "  clean        - Clean up temporary Docker resources"
	@echo "  help         - Show this help message"

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2025-08-20

### Fixed
- Docker build contexts to work with both manual builds and `docker-compose.local.yml`
- CI/CD workflows by adding missing `build` and `twine` dependencies to PyPI publishing
- Linting errors (whitespace on blank lines) in API code
- MyPy type checking issues with import statements
- Volume pathing issues in Docker configurations

### Changed
- Updated Docker image build process for better compatibility
- Improved build script reliability and error handling

### Infrastructure
- All Docker images rebuilt and pushed to DockerHub:
  - `mrilikecoding/stream_pose_ml_api:latest`
  - `mrilikecoding/stream_pose_ml_web_ui:latest`  
  - `mrilikecoding/stream_pose_ml_mlflow:latest`
- All CI/CD workflows now pass successfully
- Maintained backward compatibility with existing configurations

## [0.2.1] - 2025-06-18

### Added
- Black code formatter to CI flow
- Comprehensive testing improvements with tests for geometry, services, and jobs modules
- New make targets for improved testing workflow

### Changed
- **Documentation improvements:**
  - Updated MLFlow documentation
  - Improved startup process documentation
  - Switched to Alabaster docs theme
  - General documentation updates
- **Code refactoring:**
  - Separated API service from core package code
  - Restructured imports for better organization
  - Merged test branch with comprehensive testing improvements

### Fixed
- **MLFlow compatibility:**
  - Pinned MLFlow to version < 2.22.1 to avoid breaking changes in version 3
  - Addressed model serving compatibility (gunicorn to FastAPI transition)
  - Removed old MLFlow Dockerfile and run scripts from package code
- API CI path issues
- Moved API tests to proper directory
- Improved start/stop process with updated build scripts

### Infrastructure
- Updated build scripts for better reliability

## [0.2.0] - 2024-12-07

### Added
- **MLFlow Integration:** Complete integration of [MLFlow](https://mlflow.org/docs/latest/models.html) support
- MLFlow support in the web application
- MLFlow-related functionality in the pip package
- Flexible model deployment using MLFlow's standard toolkit

### Changed
- Enhanced model training workflow to support MLFlow's schema
- Improved compatibility with MediaPipe keypoint generation
- Expanded beyond specific ScikitLearn and XGBoost wrappers to support broader ML frameworks

### Infrastructure
- Added MLFlow container and services to the application stack

## [0.1.1] - 2023-11-20

### Added
- Build configuration improvements
- Updated build system components

### Fixed
- Various build-related issues

## [0.1.0] - 2023-11-20

### Added
- **Initial Alpha Release**
- Core StreamPoseML pip package functionality
- Parallel web application stack
- Example notebooks demonstrating initial use cases
- MediaPipe-based pose estimation capabilities
- Basic machine learning pipeline for pose classification

### Infrastructure
- Initial project structure and build system
- Docker containerization setup
- Basic CI/CD workflows

---

## Contributing

When adding new entries to this changelog:

1. **Keep entries in reverse chronological order** (newest first)
2. **Use semantic versioning** for version numbers
3. **Categorize changes** using:
   - `Added` for new features
   - `Changed` for changes in existing functionality
   - `Deprecated` for soon-to-be removed features
   - `Removed` for now removed features
   - `Fixed` for any bug fixes
   - `Security` for vulnerability fixes
   - `Infrastructure` for deployment/build changes
4. **Include relevant links** to issues, PRs, or documentation
5. **Be concise but descriptive** about what changed and why it matters

For more details on any release, see the [GitHub Releases](https://github.com/mrilikecoding/StreamPoseML/releases) page.
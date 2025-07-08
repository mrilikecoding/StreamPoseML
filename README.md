# StreamPoseML

#### An End-to-End Open-Source Web Application and Python Toolkit for Real-Time Video Pose Classification and Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Supported Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Windows%20%7C%20Linux-green)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14298482.svg)](https://doi.org/10.5281/zenodo.14298482)
[![Documentation Status](https://readthedocs.org/projects/streamposeml/badge/?version=latest)](https://streamposeml.readthedocs.io/en/latest/?badge=latest)

## Overview

StreamPoseML is an open-source toolkit for creating real-time, video-based classification applications using body pose data. It provides both a Python package and a web application to help you:

1. **Process Video Data** - Extract pose keypoints from videos using MediaPipe
2. **Build Datasets** - Merge keypoint data with annotations and generate features
3. **Train Models** - Train and evaluate machine learning models for pose classification
4. **Deploy Applications** - Run real-time classification in web browsers or Python environments

## Documentation

Full documentation is available at **[streamposeml.readthedocs.io](https://streamposeml.readthedocs.io)**

- **Getting Started Guide** - Installation and basic usage
- **API Reference** - Detailed class and method documentation
- **Workflow Tutorials** - Step-by-step instructions for common tasks
- **Web Application Guide** - Running and customizing the web application

## Components

The StreamPoseML project consists of two main parts:

1. **Python Package** (`stream_pose_ml/`)
   - Available on PyPI: `pip install stream-pose-ml` or `uv add stream-pose-ml`
   - Core tools for video processing, pose extraction, dataset creation, and model training
   - Can be used independently in your Python projects

2. **Web Application** (Docker-based)
   - React frontend for webcam capture and visualization
   - Flask API backend for model serving
   - MLflow integration for standardized model deployment
   - Ready-to-use Docker images available on DockerHub

## Quick Start

### Python Package

```bash
# Install the package
pip install stream-pose-ml
# Or with uv (recommended for development)
uv add stream-pose-ml

# Import core modules
import stream_pose_ml.jobs.process_videos_job as pv
import stream_pose_ml.jobs.build_and_format_dataset_job as data_builder
import stream_pose_ml.learning.model_builder as mb
```

### Web Application

```bash
# Clone the repository
git clone https://github.com/mrilikecoding/StreamPoseML.git
cd StreamPoseML

# Start using pre-built images
make start

# Or start with local code (development mode)
make start-dev

# When finished
make stop
```

## Key Features

- **MediaPipe Integration** - Uses [MediaPipe](https://developers.google.com/mediapipe)'s BlazePose for efficient pose detection
- **Feature Engineering** - Generates angles, distances, and normalized measurements from raw keypoints
- **Annotation Support** - Merges video keypoints with external annotation files
- **Flexible Dataset Creation** - Various segmentation strategies for time-series data
- **Model Building Utilities** - Convenience methods for training and evaluation
- **Real-time Classification** - Browser-based pose classification with webcam input
- **MLflow Integration** - Standardized model serving and deployment

## Example Use Case

StreamPoseML was built while conducting studies of Parkinson's Disease patients in dance therapy settings. This research was done with support from the [McCamish Foundation](https://parkinsons.gatech.edu/).

## Development

A comprehensive developer guide is available in the documentation. Key commands:

```bash
# Install in development mode
uv sync --extra dev

# Run tests
make test
make test-core  # Package tests only
make test-api   # API tests only

# Start application (development mode)
make start-dev

# Show all available commands
make help
```

## Publications

Research using StreamPoseML:

1. **Closed-loop Neuromotor Training System Pairing Transcutaneous Vagus Nerve Stimulation with Video-based Real-time Movement Classification**  
   [https://www.medrxiv.org/content/10.1101/2025.05.23.25327218v1](https://www.medrxiv.org/content/10.1101/2025.05.23.25327218v1)

2. **StreamPoseML: An End-to-End Open-Source Web Application and Python Toolkit for Real-Time Video Pose Classification and Machine Learning**  
   [https://joss.theoj.org/papers/10.21105/joss.06392](https://joss.theoj.org/papers/10.21105/joss.06392)

## Citing

If you use StreamPoseML in your work or research, please cite:

```bibtex
@software{streamposeml2023,
  author = {Green, Nate},
  title = {StreamPoseML: Toolkit for Real-Time Video Pose Classification},
  url = {https://github.com/mrilikecoding/StreamPoseML},
  doi = {10.5281/zenodo.14298482},
  year = {2023}
}
```

See [paper.md](paper.md) for more details.

## Contribute to StreamPoseML

We're actively seeking contributors! Whether you're fixing bugs, adding features, improving documentation, or sharing your use cases, your contribution matters.

### Ways to Contribute

- **Code**: Fix bugs, implement new features, or improve performance
- **Documentation**: Help improve or translate documentation
- **Testing**: Create tests or report bugs
- **Examples**: Share your use cases or implementation examples
- **Research**: Cite us in your research or suggest new features based on research needs

Check our [contribution guidelines](CONTRIBUTING.md) and [open issues](https://github.com/mrilikecoding/StreamPoseML/issues) to get started. New contributors are welcome - we've labeled some issues as "good first issue" to help you begin!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

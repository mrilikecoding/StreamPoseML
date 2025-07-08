Installation
============

Requirements
-----------

StreamPoseML requires:

* Python 3.10 or 3.11
* Dependencies (installed automatically with pip)

Installing StreamPoseML
----------------------

The easiest way to install StreamPoseML is via pip or uv:

.. code-block:: bash

   pip install stream-pose-ml
   # Or with uv (recommended for development)
   uv add stream-pose-ml

This will automatically install all required dependencies.

Dependencies
-----------

StreamPoseML depends on several libraries:

* mediapipe (>= 0.10.21)
* opencv-contrib-python
* numpy
* pandas
* scikit-learn
* xgboost
* mlflow (>= 2.18.0, < 2.21.0)

Development Installation
-----------------------

For development, you can install the package with development dependencies:

.. code-block:: bash

   git clone https://github.com/mrilikecoding/StreamPoseML.git
   cd StreamPoseML
   uv sync --extra dev

Verifying Installation
---------------------

To verify that StreamPoseML is installed correctly, you can import it in Python:

.. code-block:: python

   import stream_pose_ml
   print(stream_pose_ml.__version__)  # Should print the current version

Docker Installation
------------------

To run the web application components with Docker:

1. Install Docker and Docker Compose
2. Clone the repository
3. Run the application:

.. code-block:: bash

   git clone https://github.com/mrilikecoding/StreamPoseML.git
   cd StreamPoseML
   make start
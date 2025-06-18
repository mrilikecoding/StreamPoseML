Installation
============

Prerequisites
------------

To run the StreamPoseML web application, you'll need:

* Docker and Docker Compose
* Git (to clone the repository)

Quickstart Installation
----------------------

The fastest way to get started is using the pre-built Docker images:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/mrilikecoding/StreamPoseML.git
      cd StreamPoseML

2. Start the application:

   .. code-block:: bash

      make start

   This will:
   
   * Pull the necessary Docker images from Docker Hub
   * Start the containers for the API, web UI, and MLflow server
   * Launch the application in your default browser

3. When you're done, stop the application:

   .. code-block:: bash

      make stop

For debugging purposes, you can also start the application with debug output:

.. code-block:: bash

   make start-debug

Local Development Installation
-----------------------------

For developers who want to modify the web application:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/mrilikecoding/StreamPoseML.git
      cd StreamPoseML

2. Install Tilt and Minikube:
   
   - `Tilt <https://tilt.dev>`_
   - `Minikube <https://minikube.sigs.k8s.io/docs/>`_

3. Start local development:

   .. code-block:: bash

      tilt up

   This will build the containers locally and start them with live reload for development.

Manual Docker Installation
-------------------------

If you prefer to build the Docker images manually:

.. code-block:: bash

   # Build API image
   cd stream_pose_ml && docker build -t myuser/stream_pose_ml_api:latest -f Dockerfile .
   
   # Build web UI image
   cd web_ui && docker build -t myuser/stream_pose_ml_web_ui:latest -f Dockerfile .
   
   # Push images (if deploying to a registry)
   docker push myuser/stream_pose_ml_api:latest
   docker push myuser/stream_pose_ml_web_ui:latest
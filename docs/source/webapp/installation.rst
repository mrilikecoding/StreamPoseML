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
   * Use pre-built images rather than building from source code

3. When you're done, stop the application:

   .. code-block:: bash

      make stop

Additional Startup Options
^^^^^^^^^^^^^^^^^^^^^^^^^

For debugging purposes, you can start the application with debug output:

.. code-block:: bash

   make start-debug

For development purposes, you can build and run from local code:

.. code-block:: bash

   make start-dev

Local Development Installation
-----------------------------

For developers who want to modify the web application:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/mrilikecoding/StreamPoseML.git
      cd StreamPoseML

2. Install Docker and Docker Compose:
   
   - `Docker <https://docs.docker.com/get-docker/>`_
   - Docker Compose is included with Docker Desktop

3. Start local development with local code:

   .. code-block:: bash

      make start-dev

   This will:
   
   * Build containers from the local source code
   * Hot-reload the API code when you make changes
   * Mount your local package code into the container
   * Provide a development environment for making changes to the code

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
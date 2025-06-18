Web Application
===============

StreamPoseML includes a full-featured web application for real-time pose classification. The web application consists of:

* A React-based frontend for webcam capture and visualization
* A Flask-based API for classification
* **Built-in MLflow integration** for seamless deployment of trained models

This section covers the installation, configuration, and usage of the StreamPoseML web application.

MLflow Integration
------------------

A standout feature of the StreamPoseML web application is its **direct integration with MLflow** for model deployment. This provides several advantages:

* **Standardized Model Serving**: Deploy models tracked with MLflow without extra conversion steps
* **Version Management**: Easily switch between different model versions
* **Metadata Tracking**: Access model parameters, metrics, and artifacts
* **Consistent API**: Use the same interface for different model types

The web application includes a dedicated MLflow container that handles model loading and prediction requests, making it easy to deploy your trained models for real-time classification.

.. toctree::
   :maxdepth: 2

   installation
   usage
Welcome to StreamPoseML Documentation
=====================================

StreamPoseML is an open-source, end-to-end toolkit for creating realtime, video-based classification experiments that rely on using labeled data alongside captured body keypoint / pose data.

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/platforms-macOS%20%7C%20Windows%20%7C%20Linux-green
   :alt: Supported Platforms

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.14298482.svg
   :target: https://doi.org/10.5281/zenodo.14298482
   :alt: DOI

StreamPoseML has two main components:

1. **Python Package**: A library for pose extraction, dataset creation, and model training
2. **Web Application**: A full-featured web app for real-time pose classification

This documentation is organized by use case. If you're primarily interested in:

- Using the **Python Package** for data processing and model training, focus on the "Getting Started", "Workflows", "StreamPoseML Package Reference", and "Examples" sections.
- Deploying the **Web Application** for real-time classification, focus on the "Web Application" section.

Features
--------

* Extract pose keypoints from videos using MediaPipe (BlazePose)
* Build datasets by merging keypoint data with annotations
* Generate derived features (angles, distances, etc.) from raw keypoints
* Train and evaluate machine learning models for pose classification
* Deploy models for real-time classification from video streams
* Integrate with web applications for browser-based deployment

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents
   
   guide/index
   workflows/index
   api/index
   webapp/index
   examples/index

Contributing
------------

.. toctree::
   :maxdepth: 2
   :caption: Contributing
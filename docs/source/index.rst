Welcome to StreamPoseML Documentation
=====================================

Turning human movement into machine learning insights
----------------------------------------------------

StreamPoseML is an open-source toolkit for creating real-time, video-based movement classification applications. Whether you're a researcher studying movement patterns, a developer building interactive applications, or an artist exploring interactive technology, StreamPoseML helps you transform video of human movement into actions.

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/platforms-macOS%20%7C%20Windows%20%7C%20Linux-green
   :alt: Supported Platforms

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.14298482.svg
   :target: https://doi.org/10.5281/zenodo.14298482
   :alt: DOI

Choose Your Path
---------------

.. raw:: html

   <div style="display: flex; justify-content: space-between; margin: 30px 0;">
     <div style="width: 30%; padding: 20px; background-color: #f8f9fa; border-radius: 5px;">
       <h3>I want to get started quickly</h3>
       <p>Jump right in with a quick-start guide to see results in minutes.</p>
       <a href="guide/quickstart.html" style="font-weight: bold;">Quick Start Guide →</a>
     </div>
     <div style="width: 30%; padding: 20px; background-color: #f8f9fa; border-radius: 5px;">
       <h3>I want to understand concepts</h3>
       <p>Learn core concepts behind pose detection and feature engineering.</p>
       <a href="guide/concepts.html" style="font-weight: bold;">Core Concepts →</a>
     </div>
     <div style="width: 30%; padding: 20px; background-color: #f8f9fa; border-radius: 5px;">
       <h3>I'm ready to build something</h3>
       <p>Follow the tutorials to build working applications.</p>
       <a href="examples/notebook_walkthrough.html" style="font-weight: bold;">Complete Example →</a>
     </div>
   </div>

StreamPoseML gives you two powerful components:

1. **Python Package**: A flexible toolkit for pose extraction, dataset creation, and model training that you can use in your Python projects
2. **Web Application**: A ready-to-use application for real-time pose classification from webcams or video files

Common Tasks
-----------

* **Process videos** to extract pose keypoints :doc:`Learn how → <workflows/video_processing>`
* **Create labeled datasets** for machine learning :doc:`Guide → <guide/annotations>`
* **Train models** to classify movements :doc:`Example → <examples/notebook_walkthrough>`
* **Deploy** for real-time classification :doc:`Web App Guide → <webapp/usage>`

Key Features
-----------

* **Powerful Pose Detection**: Extract accurate body keypoints from videos using MediaPipe's BlazePose
* **Smart Feature Engineering**: Automatically calculate angles, distances, and other features from raw keypoints
* **Flexible Dataset Creation**: Various tools for creating and transforming machine learning datasets
* **Streamlined Model Building**: Train, evaluate, and deploy classification models with minimal code
* **Real-time Classification**: Process live video streams for immediate feedback
* **Web Integration**: Deploy models in browser-based applications

Documentation Structure
---------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   
   guide/index
   guide/quickstart
   guide/concepts
   guide/installation

.. toctree::
   :maxdepth: 2
   :caption: Tutorials & Workflows
   
   workflows/index
   workflows/video_processing
   examples/notebook_walkthrough

.. toctree::
   :maxdepth: 2
   :caption: Reference
   
   api/index
   api/clients
   
.. toctree::
   :maxdepth: 2
   :caption: Web Application
   
   webapp/index
   webapp/installation
   webapp/usage

.. toctree::
   :maxdepth: 2
   :caption: Development
   
   development/contributing
   development/development_workflow

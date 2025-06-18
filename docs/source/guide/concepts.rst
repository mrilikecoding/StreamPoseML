Core Concepts
=============

Pose Detection and Keypoint Extraction
-------------------------------------

Introduction to Pose Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

StreamPoseML uses MediaPipe's BlazePose model to detect and track human poses in video. The pose detection pipeline extracts keypoints (or landmarks) from each frame, which represent the positions of various body joints like shoulders, elbows, wrists, hips, knees, and ankles.

Keypoint Representation
~~~~~~~~~~~~~~~~~~~~~

Each keypoint contains:

* 3D coordinates (x, y, z) in pixel space
* Normalized coordinates relative to a reference point (typically hip width)
* Visibility and confidence scores

These keypoints are organized into a ``BlazePoseFrame`` for each video frame, which can then be combined into a ``BlazePoseSequence`` representing a continuous motion.

BlazePose Keypoints
~~~~~~~~~~~~~~~~~

The system uses MediaPipe's standard keypoints which include:

* Facial landmarks (nose, eyes, ears)
* Upper body landmarks (shoulders, elbows, wrists, hands)
* Lower body landmarks (hips, knees, ankles, feet)

OpenPose Compatibility
~~~~~~~~~~~~~~~~~~~~

StreamPoseML includes utilities for transforming BlazePose keypoints to formats compatible with OpenPose's Body-25 model, allowing for compatibility with datasets and models trained on OpenPose data.

Sequence Processing
-----------------

Frame Sequences
~~~~~~~~~~~~~

Video data is processed as sequences of frames, where each frame contains pose keypoints. A ``BlazePoseSequence`` represents a continuous segment of frames with associated metadata.

Key concepts:

* **Frame Window**: Number of consecutive frames considered as a single sequence
* **Frame Overlap**: Number of overlapping frames between consecutive sequences (for MLFlowClient)
* **Sequence Generation**: Process of converting raw frames into structured sequences

Feature Engineering
~~~~~~~~~~~~~~~~

From raw keypoints, StreamPoseML can compute various derived features:

* **Angles**: Angular relationships between body segments (e.g., elbow angle)
* **Distances**: Spatial relationships between keypoints
* **Vectors**: Directional relationships between joints
* **Normalized Features**: Features scaled relative to body proportions

These features enhance the discriminative power of the data for classification tasks.

Dataset Creation
--------------

Dataset Structure
~~~~~~~~~~~~~~

StreamPoseML datasets combine keypoint data with annotations/labels. The typical structure includes:

* **Sequences**: Time series of pose keypoints
* **Annotations**: Labels with start/end frame information
* **Features**: Raw and derived features from keypoints
* **Metadata**: Additional information about the sequences

Annotation Integration
~~~~~~~~~~~~~~~~~~~

Annotations are typically provided as JSON files with:

* Label information
* Start and end frames
* Additional metadata

StreamPoseML provides tools to merge these annotations with extracted keypoint data.

Segmentation Strategies
~~~~~~~~~~~~~~~~~~~~

When building datasets, different segmentation strategies can be applied:

* **None**: Raw frame-by-frame data
* **Flatten into columns**: Temporal data represented as separate columns
* **Window-based**: Fixed-size windows with potential overlap
* **Custom**: User-defined segmentation logic

Model Training
------------

Dataset Preparation
~~~~~~~~~~~~~~~~

Before training, datasets typically undergo:

* Feature selection
* Normalization
* Train/test splitting
* Handling class imbalance (if necessary)

Model Types
~~~~~~~~~

StreamPoseML is agnostic to the model type and supports:

* Traditional ML models (Random Forest, XGBoost)
* Deep learning models (via integration with external libraries)
* Custom model architectures

Evaluation
~~~~~~~~

Model evaluation considers:

* Accuracy metrics
* Precision and recall
* F1 score
* Confusion matrices
* Cross-validation results

Real-time Classification
---------------------

Pipeline Structure
~~~~~~~~~~~~~~

The real-time classification pipeline involves:

1. Capturing video frames
2. Extracting pose keypoints
3. Maintaining a buffer of recent frames
4. Computing features
5. Applying the trained model
6. Producing classification results

Integration Models
~~~~~~~~~~~~~~~

StreamPoseML provides two main integration models:

* **Direct Integration** via StreamPoseClient
* **MLflow-based Integration** via MLFlowClient

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~

Real-time classification requires:

* Efficient frame processing
* Optimized feature computation
* Fast model inference
* Proper buffer management to balance accuracy and latency
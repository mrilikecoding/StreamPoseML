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

Here's how keypoint data is structured in the codebase:

.. code-block:: python

   # Example of keypoint data structure from frame files
   {
     "sequence_id": 1234567890,
     "sequence_source": "mediapipe",
     "frame_number": 1,
     "image_dimensions": {"height": 1080, "width": 1920},
     "joint_positions": {
       "nose": {
         "x": 960.5,
         "y": 540.2,
         "z": 0.05,
         "x_normalized": 0.5,
         "y_normalized": 0.5,
         "z_normalized": 0.05
       },
       "left_shoulder": {
         // ...
       }
     }
   }

   # From stream_pose_ml/stream_pose_client.py
   # This is how frames are stored in the StreamPoseClient
   frame_data = {
       "sequence_id": None,
       "sequence_source": "web",
       "frame_number": None,
       "image_dimensions": None,
       "joint_positions": self.mpc.serialize_pose_landmarks(
           pose_landmarks=list(keypoint_results.pose_landmarks.landmark)
       )
   }

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

Here's how frame sequences are created in the code:

.. code-block:: python

   # From stream_pose_ml/stream_pose_client.py
   # Creating a BlazePoseSequence from frame data
   def run_frame_pipeline(self, image: np.ndarray):
       results = self.get_keypoints(image)
       current_frames = self.update_frame_data(results)
       if len(current_frames) == self.frame_window:
           sequence = BlazePoseSequence(
               name=f"sequence-{time.time_ns()}",
               sequence=list(current_frames),
               include_geometry=True,
           ).generate_blaze_pose_frames_from_sequence()
           sequence_data = BlazePoseSequenceSerializer().serialize(sequence)
           # Get columns from model
           columns = self.model.model_data["X_test"].columns.tolist()
           data, meta = self.transformer.transform(data=sequence_data, columns=columns)
           self.current_classification = bool(self.model.predict(data=data)[0])

Feature Engineering
~~~~~~~~~~~~~~~~

From raw keypoints, StreamPoseML can compute various derived features:

* **Angles**: Angular relationships between body segments (e.g., elbow angle)
* **Distances**: Spatial relationships between keypoints
* **Vectors**: Directional relationships between joints
* **Normalized Features**: Features scaled relative to body proportions

These features enhance the discriminative power of the data for classification tasks.

Here's how feature selection works when formatting a dataset:

.. code-block:: python

   # From stream_pose_ml/jobs/build_and_format_dataset_job.py
   # Selecting which features to include in the dataset
   formatted_dataset = db.format_dataset(
       dataset=dataset,
       pool_frame_data_by_clip=False,  # Whether to pool features across frames
       decimal_precision=4,            # Precision for numerical values
       include_unlabeled_data=True,    # Include frames without labels
       include_angles=True,            # Include angle features
       include_distances=True,         # Include distance features
       include_normalized=True,        # Include normalized features
       include_joints=False,           # Include raw joint positions
       include_z_axis=False,           # Include z-axis data
       segmentation_strategy="flatten_on_example",  # How to segment the data
       segmentation_splitter_label="step_type",     # Label to use for segmentation
       segmentation_window=10,                     # Window size
       segmentation_window_label="weight_transfer_type" # Label for window
   )

Dataset Creation
--------------

Dataset Structure
~~~~~~~~~~~~~~

StreamPoseML datasets combine keypoint data with annotations/labels. The typical structure includes:

* **Sequences**: Time series of pose keypoints
* **Annotations**: Labels with start/end frame information
* **Features**: Raw and derived features from keypoints
* **Metadata**: Additional information about the sequences

Here's how datasets are built in the code:

.. code-block:: python

   # From stream_pose_ml/jobs/build_and_format_dataset_job.py
   # Building a dataset from annotation files and processed sequences
   dataset = db.build_dataset_from_data_files(
       annotations_data_directory='/path/to/annotations',
       sequence_data_directory='/path/to/sequences',
       limit=None,  # Optional: limit the number of files processed
   )
   
   # Internally, this creates a Dataset object with structured data:
   dataset = Dataset(
       all_frames=annotated_video_data["all_frames"],
       labeled_frames=annotated_video_data["labeled_frames"],
       unlabeled_frames=annotated_video_data["unlabeled_frames"],
   )

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

Here are examples of different segmentation strategies from the example notebook:

.. code-block:: python

   # Raw frame-by-frame data (no segmentation)
   formatted_dataset = db.format_dataset(
       dataset=dataset,
       include_angles=True,
       include_distances=True,
       segmentation_strategy="none"
   )
   
   # Flatten columns over window
   formatted_dataset = db.format_dataset(
       dataset=dataset,
       include_angles=True,
       include_distances=True,
       segmentation_strategy="flatten_into_columns",
       segmentation_splitter_label="step_type",
       segmentation_window=10,
       segmentation_window_label="weight_transfer_type"
   )
   
   # Flatten on example with window
   formatted_dataset = db.format_dataset(
       dataset=dataset,
       include_angles=True,
       include_distances=True,
       segmentation_strategy="flatten_on_example",
       segmentation_splitter_label="step_type",
       segmentation_window=10,
       segmentation_window_label="weight_transfer_type"
   )

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

Here's how to train different model types using the codebase:

.. code-block:: python

   # From the example notebook - Training a Gradient Boost model
   from stream_pose_ml.learning import model_builder as mb
   
   # Mapping string categories to numerical values
   value_map = {
       "weight_transfer_type": {
           "Failure Weight Transfer": 0,
           "Successful Weight Transfer": 1,
       }
   }
   # Columns to drop from the dataset
   drop_list = ["video_id", "step_frame_id", "frame_number", "step_type"]
   
   model_builder = mb.ModelBuilder()
   
   # Load and prepare dataset
   model_builder.load_and_prep_dataset_from_csv(
       path="path/to/dataset.csv",
       target="weight_transfer_type",
       value_map=value_map,
       column_whitelist=[],  # Empty means use all columns not in drop_list
       drop_list=drop_list,
   )
   
   # Configure train/test split
   model_builder.set_train_test_split(
       balance_off_target=True,
       upsample_minority=True,
       downsample_majority=False,
       use_SMOTE=False,
       random_state=40002,
   )
   
   # Train gradient boost model
   model_builder.train_gradient_boost()
   
   # Evaluate the model
   model_builder.evaluate_model()
   
   # Train random forest model with hyperparameter tuning
   param_dist = {
       "n_estimators": [20, 50, 100, 200],
       "max_depth": 9,
       "max_leaf_nodes": 63,
   }
   
   model_builder.train_random_forest(
       use_random_search=True, 
       params=param_dist, 
       iterations=50, 
       random_state=123
   )

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

Here's the implementation of the real-time classification pipeline from the codebase:

.. code-block:: python

   # From stream_pose_ml/stream_pose_client.py
   # The StreamPoseClient implements the real-time classification pipeline
   
   def run_frame_pipeline(self, image: np.ndarray):
       # Step 1 & 2: Process the image and extract keypoints
       results = self.get_keypoints(image)
       
       # Step 3: Update the frame buffer with new keypoints
       current_frames = self.update_frame_data(results)
       
       # When buffer is full, perform classification
       if len(current_frames) == self.frame_window:
           # Create a sequence from buffered frames
           sequence = BlazePoseSequence(
               name=f"sequence-{time.time_ns()}",
               sequence=list(current_frames),
               include_geometry=True,
           ).generate_blaze_pose_frames_from_sequence()
           
           # Serialize the sequence
           sequence_data = BlazePoseSequenceSerializer().serialize(sequence)
           
           # Step 4: Transform raw data to features expected by model
           columns = self.model.model_data["X_test"].columns.tolist()
           data, meta = self.transformer.transform(data=sequence_data, columns=columns)
           
           # Step 5 & 6: Apply the model and get classification result
           self.current_classification = bool(self.model.predict(data=data)[0])
           
       return True

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
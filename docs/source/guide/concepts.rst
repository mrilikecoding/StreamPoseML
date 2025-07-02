Core Concepts
=============

Understanding how StreamPoseML works
----------------------------------

This guide explains the key concepts behind StreamPoseML in simple, clear terms. Whether you're new to pose estimation or an experienced ML practitioner, this will help you understand how the system works.

.. figure:: /_static/logo.png
   :align: center
   :alt: StreamPoseML workflow visualization
   
   *StreamPoseML transforms videos of people moving into useful machine learning insights*

Pose Detection and Keypoint Extraction
-------------------------------------

How computers see human movement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

StreamPoseML detects how people move in videos. The system uses MediaPipe's BlazePose technology to detect and track human poses. Think of it like a digital skeleton that follows along with a person's movements in video.

Understanding Keypoints: The Digital Skeleton
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When StreamPoseML processes a video, it identifies key points on the human body in each frame. Think of these as digital push pins placed at important joints and body parts.

Each keypoint contains valuable information:

* **Position**: Where is this body part in 3D space? (x, y, z coordinates)
* **Normalized position**: Adjusted coordinates that work regardless of image size or person's distance from camera
* **Confidence**: How sure is the system that it correctly identified this body part?

For each video frame, these keypoints are wrapped into a neat package called a ``BlazePoseFrame``. When you put multiple frames together in sequence, you get a ``BlazePoseSequence`` - essentially a digital record of how someone moved over time.

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

The Digital Body Map: What Points Are Tracked
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

StreamPoseML tracks a comprehensive set of points on the human body using MediaPipe's technology:

* **Face**: Nose, eyes, ears, and mouth
* **Upper Body**: Shoulders, elbows, wrists, hands, and fingers
* **Lower Body**: Hips, knees, ankles, and feet

Think of it as placing motion-tracking dots on an actor, but done entirely through video analysis - no special suits or equipment needed!

**Compatibility with Other Systems**: If you've worked with OpenPose before (another popular pose estimation system), don't worry! StreamPoseML can convert its data format to be compatible with OpenPose's Body-25 model, making it easy to work with existing datasets and models.

From Frames to Movement: Sequence Processing
----------------------------------------

Capturing Motion Over Time
~~~~~~~~~~~~~~~~~~~~~~~

Movement happens over time, not in a single snapshot. That's why StreamPoseML processes videos as sequences of frames. Imagine flipping through a flipbook animation - each page shows a slightly different position, and together they create fluid motion.

In StreamPoseML, a ``BlazePoseSequence`` represents a continuous segment of movement across multiple video frames. This is crucial for analyzing dynamic movements like dance steps, sports techniques, or rehabilitation exercises.

Important concepts to understand:

* **Frame Window**: How many consecutive frames are grouped together as one movement unit (like analyzing 30 frames = 1 second of video)
* **Frame Overlap**: How many frames are shared between consecutive windows (helps create smoother analysis)
* **Sequence Generation**: The process that transforms individual frame data into meaningful sequences

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

Making Movement Measurable: Feature Engineering
------------------------------------------

StreamPoseML calculates features from raw keypoint positions to identify movement patterns. This process, called feature engineering, transforms raw data points into insightful metrics about how the body is moving.

StreamPoseML automatically calculates these types of features:

* **Angles**: How bent is an elbow or knee? What's the angle between torso and arm?
* **Distances**: How far apart are the hands? What's the distance from foot to hip?
* **Vectors**: In which direction is the arm moving? What's the relationship between head and shoulder movement?
* **Normalized Features**: Measurements that work regardless of the person's size or distance from camera

These calculated features are what make machine learning models truly powerful. For example, the difference between a correct and incorrect dance step might be detected in the angle of a knee bend or the relationship between torso angle and arm extension.

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

Building Your Movement Library: Dataset Creation
--------------------------------------------

Building Movement Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training movement recognition models requires multiple labeled examples. StreamPoseML helps you build rich datasets that combine movement data with meaningful labels.

A StreamPoseML dataset contains:

* **Movement Sequences**: Time-ordered series of keypoints showing how people moved
* **Labels/Annotations**: Information about what each movement represents ("correct form", "exercise type A", etc.)
* **Calculated Features**: All those angles, distances, and other measurements we talked about
* **Context Information**: Additional details about the video source, recording conditions, etc.

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

Adding Meaning: Annotation Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To make sense of movement data, we need to label what's happening in the video. For example, which frames show a "correct dance step" versus an "incorrect step"?

StreamPoseML works with annotation files (typically in JSON format) that contain:

* **Labels**: What movement is being performed?
* **Timing Information**: Which frames contain this movement? (start/end points)
* **Additional Context**: Any other relevant information about the movement

The system seamlessly merges these human-provided annotations with the computer-detected keypoints, creating a rich dataset that connects movements with their meanings.

Organizing Movement Data: Segmentation Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Movements happen over time, so how should we package this time-based data for machine learning? StreamPoseML offers several approaches:

* **Frame-by-Frame**: Each individual frame is treated as a separate data point (like analyzing a single snapshot)
* **Flattened Time Windows**: Multiple frames are combined into a single row of data (like watching 10 frames at once)
* **Sliding Windows**: Overlapping segments of frames (like a moving spotlight tracking through time)
* **Custom Segmentation**: Your own approach to organizing the time-series data

Choosing the right segmentation strategy depends on your goals. For instance, recognizing a dance step might require looking at 30 consecutive frames together, while detecting a fall might need a different approach.

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

Teaching Computers to Recognize Movements: Model Training
--------------------------------------------------

Preparing Your Data for Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have your movement dataset, there are a few key steps to prepare it for machine learning:

* **Feature Selection**: Choosing which measurements are most important (Do we need all 33 angles? Or just the knee and elbow angles?)
* **Normalization**: Adjusting values to comparable scales (so height differences between people don't confuse the model)
* **Train/Test Splitting**: Setting aside some data to evaluate how well the model generalizes
* **Handling Imbalanced Data**: Making sure the model sees enough examples of rare movements

Choosing Your Learning Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~

StreamPoseML is flexible about what kind of machine learning models you use. It works well with:

* **Traditional ML Models**: Fast, interpretable models like Random Forest and Gradient Boosting that work well for many movement classification tasks
* **Deep Learning Models**: More complex neural networks for challenging movement patterns
* **Your Custom Models**: If you've developed your own special approach

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

How Good Is Your Model? Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you've trained your model, you need to know how well it performs. StreamPoseML provides comprehensive evaluation tools that consider multiple aspects of performance:

* **Accuracy**: Overall percentage of correct predictions
* **Precision and Recall**: Balancing between false positives and false negatives
* **F1 Score**: Harmonic mean of precision and recall
* **Confusion Matrix**: Detailed breakdown of prediction successes and errors by class
* **Cross-Validation**: Testing performance across different subsets of your data

These metrics help you refine your approach and ensure your movement classifier is reliable before deployment.

Putting It All Together: Real-time Classification
---------------------------------------------

How Live Classification Works
~~~~~~~~~~~~~~~~~~~~~~~~~

The magic happens when StreamPoseML classifies movements in real time. Here's the process that runs continuously as someone moves in front of a camera:

1. **Frame Capture**: The system continuously grabs video frames from a camera feed
2. **Pose Detection**: For each frame, it detects the person and their pose keypoints
3. **Sequence Management**: It maintains a rolling window of recent frames (e.g., the last 30 frames)
4. **Feature Calculation**: It computes angles, distances, and other features from the keypoints
5. **Model Prediction**: Your trained model examines these features and makes a classification
6. **Result Delivery**: Classification results are returned for immediate feedback

This entire pipeline runs many times per second, providing smooth, responsive feedback about the movements being performed.

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

Integrating with Your Applications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

StreamPoseML gives you two primary ways to integrate movement classification into your own applications:

* **Direct Integration**: Using the `StreamPoseClient` class for simple, self-contained applications
* **MLflow Integration**: Using the `MLFlowClient` for advanced, scalable deployment with model versioning

The direct approach is simpler, while MLflow integration provides more robust model management features.

Keeping Things Fast: Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Real-time movement classification requires balancing accuracy with speed. StreamPoseML optimizes for:

* **Efficient Video Processing**: Quickly extracting pose information from frames
* **Smart Feature Computation**: Calculating only the features needed for your specific models
* **Fast Model Inference**: Ensuring predictions happen quickly enough for real-time feedback
* **Intelligent Frame Management**: Finding the right balance between historical context and responsiveness

By carefully considering these factors, StreamPoseML enables smooth, responsive real-time classification even on modest hardware.

Where to Go From Here
------------------

Now that you understand the core concepts, you're ready to start working with StreamPoseML! Check out:

* :doc:`../workflows/video_processing` - Step-by-step guide for processing videos
* :doc:`../examples/notebook_walkthrough` - Complete example workflow from video to classification
* :doc:`../api/clients` - Details on integrating StreamPoseML into your applications

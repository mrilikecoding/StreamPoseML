Video Processing Workflow
======================

This guide explains how to use StreamPoseML to process video files, extract pose keypoints, and prepare the data for subsequent analysis and model training.

Overview
--------

The video processing workflow in StreamPoseML consists of:

1. Loading source videos
2. Processing each video frame to extract pose keypoints using MediaPipe
3. Organizing keypoints into frames and sequences
4. Saving the processed data for later use

Step 1: Import Required Modules
------------------------------

.. code-block:: python

   import stream_pose_ml.jobs.process_videos_job as pv

Step 2: Initialize the ProcessVideosJob
--------------------------------------

The ``ProcessVideosJob`` class manages the video processing pipeline:

.. code-block:: python

   job = pv.ProcessVideosJob()

Step 3: Process Videos
--------------------

Use the ``process_videos`` method to extract keypoints from a directory of videos:

.. code-block:: python

   result = job.process_videos(
       src_videos_path='/path/to/source/videos',
       output_keypoints_data_path='/path/to/output/frame/keypoints',
       output_sequence_data_path='/path/to/output/video/sequences',
       write_keypoints_to_file=True,
       write_serialized_sequence_to_file=True,
       limit=None,
       configuration={},
       preprocess_video=True,
       return_output=False
   )

Parameters Explained
^^^^^^^^^^^^^^^^^^

- **src_videos_path** (str): Path to the directory containing source videos.
- **output_keypoints_data_path** (str): Path where individual frame keypoints will be saved.
- **output_sequence_data_path** (str): Path where serialized sequences will be saved.
- **write_keypoints_to_file** (bool): Whether to save individual keypoints to files.
- **write_serialized_sequence_to_file** (bool): Whether to save sequences to files.
- **limit** (int, optional): Limit the number of videos to process.
- **configuration** (dict, optional): MediaPipe configuration settings.
- **preprocess_video** (bool): Apply preprocessing to videos (contrast enhancement, etc.).
- **return_output** (bool): Return the processed data instead of saving to files.

Step 4: Output File Structure
---------------------------

After processing, the output directories will contain:

Keypoints Directory Structure
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

   output_keypoints_data_path/
   ├── video1_name/
   │   ├── frame_0001.json
   │   ├── frame_0002.json
   │   └── ...
   ├── video2_name/
   │   ├── frame_0001.json
   │   └── ...
   └── ...

Each frame JSON file contains keypoint data in the format:

.. code-block:: json

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
       },
       // Additional joints
     }
   }

Sequences Directory Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

   output_sequence_data_path/
   ├── video1_name.json
   ├── video2_name.json
   └── ...

Each sequence JSON file contains the entire video's frame data in a structured format that can be directly used for dataset creation.

Step 5: Processing Single Videos
------------------------------

For more control, you can process individual videos:

.. code-block:: python

   from stream_pose_ml.jobs.process_video_job import ProcessVideoJob

   # From stream_pose_ml/jobs/process_video_job.py
   # This shows how to process a single video
   result = ProcessVideoJob.process_video(
       input_filename='example_video.mp4',
       video_input_path='/path/to/videos',
       output_keypoint_data_path='/path/to/output/keypoints',
       output_sequence_data_path='/path/to/output/sequences',
       write_keypoints_to_file=True,
       write_serialized_sequence_to_file=True,
       configuration={},
       preprocess_video=True
   )

Advanced Configuration
--------------------

MediaPipe Configuration
^^^^^^^^^^^^^^^^^^^^

You can customize MediaPipe's pose detection parameters:

.. code-block:: python

   configuration = {
       'min_detection_confidence': 0.7,
       'min_tracking_confidence': 0.7,
       'model_complexity': 2,  # 0, 1, or 2 (highest accuracy)
       'smooth_landmarks': True
   }

   job.process_videos(
       # ... other parameters ...
       configuration=configuration
   )

Video Preprocessing
^^^^^^^^^^^^^^^^

By default, StreamPoseML applies contrast enhancement to improve keypoint detection. You can disable this if your videos are already well-processed:

.. code-block:: python

   job.process_videos(
       # ... other parameters ...
       preprocess_video=False
   )

Error Handling
------------

The processing job will skip videos that cannot be processed and continue with the rest. It will log errors for troubleshooting. To ensure all videos are processed correctly:

1. Check video formats (MP4, WebM, AVI are supported)
2. Ensure videos are readable and not corrupted
3. Verify sufficient disk space for output files
4. Check that MediaPipe can detect poses in your video content

Performance Considerations
------------------------

Video processing can be computationally intensive. Consider:

- Processing in batches when dealing with many videos
- Using the ``limit`` parameter for testing before full processing
- Ensuring sufficient RAM for large videos
- Using a machine with GPU acceleration for faster processing

Next Steps
----------

After processing videos, you'll typically want to:

1. Create :doc:`../guide/annotations` for your videos
2. Use annotations and sequences to build a dataset
3. Train a model on the resulting dataset

See :doc:`../guide/annotations` for details on the annotation format and how to create datasets.
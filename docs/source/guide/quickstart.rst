Quick Start Guide
=================

This guide will help you get started with StreamPoseML quickly.

Basic Usage
-----------

StreamPoseML provides tools for processing videos, extracting pose keypoints, building datasets, and training models for pose classification.

.. code-block:: python

   import stream_pose_ml

   # Main client for real-time classification
   from stream_pose_ml import StreamPoseClient

   # Client for MLflow integration
   from stream_pose_ml import MLFlowClient

   # Jobs for video processing and dataset creation
   import stream_pose_ml.jobs.process_videos_job as pv
   import stream_pose_ml.jobs.build_and_format_dataset_job as data_builder

Video Processing
----------------

Extract pose keypoints from videos:

.. code-block:: python

   # Process videos and extract pose keypoints
   pv.ProcessVideosJob().process_videos(
       src_videos_path='/path/to/source/videos',
       output_keypoints_data_path='/path/to/output/frame/keypoints',
       output_sequence_data_path='/path/to/output/video/sequences',
       write_keypoints_to_file=True,
       write_serialized_sequence_to_file=True
   )

Creating a Dataset
------------------

Merge extracted keypoints with annotations:

.. code-block:: python

   # Initialize the dataset builder
   db = data_builder.BuildAndFormatDatasetJob()

   # Build dataset from annotations and sequence data
   dataset = db.build_dataset_from_data_files(
       annotations_data_directory='/path/to/annotations',
       sequence_data_directory='/path/to/sequences'
   )

   # Format the dataset with desired features
   formatted_dataset = db.format_dataset(
       dataset=dataset,
       include_angles=True,
       include_distances=True,
       include_normalized=True,
       segmentation_strategy="none"
   )

   # Save the formatted dataset
   db.write_dataset_to_csv(
       csv_location='/path/to/output',
       formatted_dataset=formatted_dataset,
       filename="my_dataset"
   )

Real-time Classification
------------------------

Use a trained model for real-time pose classification:

.. code-block:: python

   from stream_pose_ml import StreamPoseClient
   from stream_pose_ml.blaze_pose.mediapipe_client import MediaPipeClient
   from stream_pose_ml.learning.trained_model import TrainedModel
   from stream_pose_ml.learning.sequence_transformer import SequenceTransformer

   # Initialize components
   mpc = MediaPipeClient()
   model = TrainedModel()  # Load your trained model here
   transformer = SequenceTransformer()

   # Create a StreamPoseClient for real-time classification
   client = StreamPoseClient(
       frame_window=30,
       mediapipe_client_instance=mpc,
       trained_model=model,
       data_transformer=transformer
   )

   # Process a video frame
   import cv2
   image = cv2.imread('path/to/image.jpg')
   client.run_frame_pipeline(image)

   # Get the classification result
   classification = client.current_classification

Next Steps
----------

For more detailed information, see:

* :doc:`../workflows/video_processing` - Detailed video processing workflow
* :doc:`../guide/concepts` - Core concepts and dataset creation
* :doc:`../api/clients` - API documentation for model training and usage
* :doc:`../webapp/usage` - Web application for real-time classification
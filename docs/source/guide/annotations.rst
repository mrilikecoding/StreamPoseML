Annotation Format and Usage
========================

Creating and using proper annotations is a crucial step in building effective pose classification models with StreamPoseML. This guide explains the annotation format, structure, and how annotations are used to create machine learning datasets.

What Are Annotations?
-------------------

In StreamPoseML, annotations are JSON files that label segments of video data with specific categories or classes. These labels are used to:

1. Identify meaningful actions or poses in videos
2. Provide ground truth for supervised machine learning
3. Associate pose keypoint data with specific labels for training

Annotation File Format
--------------------

Annotations in StreamPoseML use a structured JSON format. Each annotation file corresponds to a video and contains:

.. code-block:: json

    {
      "name": "example_video.mp4",
      "annotations": [
        {
          "label": "Right Step",
          "metadata": {
            "system": {
              "startTime": 0.233,
              "endTime": 1.635,
              "frame": 7,
              "endFrame": 49
            }
          }
        },
        {
          "label": "Successful Weight Transfer",
          "metadata": {
            "system": {
              "startTime": 0.233,
              "endTime": 1.635,
              "frame": 7,
              "endFrame": 49
            }
          }
        }
        // Additional annotations...
      ],
      "annotationsCount": 12,
      "annotated": true
    }

Key components:

- **name**: The name of the video file this annotation corresponds to
- **annotations**: An array of individual annotations
- **label**: The class or category of the annotation (e.g., "Right Step")
- **metadata.system.frame**: The starting frame number
- **metadata.system.endFrame**: The ending frame number
- **metadata.system.startTime**: The starting timestamp (in seconds)
- **metadata.system.endTime**: The ending timestamp (in seconds)

Annotation Structure Configuration
--------------------------------

StreamPoseML uses a configuration schema to define how annotations are structured and interpreted. This is typically defined in a ``config.yml`` file:

.. code-block:: yaml

    annotation_schema:
      annotations_key: "annotations"  # The key containing the list of annotations
      annotation_fields:
        label: label  # The field containing the class/label
        start_frame: metadata.system.frame  # Starting frame number
        end_frame: metadata.system.endFrame  # Ending frame number
      label_class_mapping:  # Maps labels to categories
        Left Step: step_type
        Right Step: step_type
        Successful Weight Transfer: weight_transfer_type
        Failure Weight Transfer: weight_transfer_type

This schema:

1. Defines where to find annotations in the JSON file
2. Specifies which fields contain important information
3. Maps specific labels to higher-level categories

Multiple Labels per Frame
-----------------------

A key feature of StreamPoseML's annotation system is the ability to assign multiple labels to the same video segment. For example, a segment might be labeled as both:

- "Right Step" (step_type)
- "Successful Weight Transfer" (weight_transfer_type)

This allows for multi-label classification and more nuanced analysis of pose data.

Creating Annotations
------------------

While StreamPoseML doesn't include built-in annotation tools, you can create annotations using:

1. Video annotation tools like CVAT, LabelStudio, or VGG VIA
2. Custom scripts that generate JSON files in the required format
3. Manual creation of JSON annotation files

Ensure your annotation files:

- Use the proper JSON structure
- Have correct frame numbers that match your videos
- Use consistent labels that align with your classification goals
- Are saved with filenames that correspond to your video files

How Annotations are Used to Build Datasets
----------------------------------------

When you create a dataset using the ``BuildAndFormatDatasetJob``, the following process occurs:

1. **Mapping**: The system maps between annotation files and sequence/video files based on filenames

   .. code-block:: python

      # The job looks for annotation files that match sequence files
      dataset = db.build_dataset_from_data_files(
          annotations_data_directory='/path/to/annotations',
          sequence_data_directory='/path/to/sequences'
      )

2. **Transformation**: The ``AnnotationTransformerService`` processes the annotations:
   - Extracts labels and frame ranges
   - Associates each frame with appropriate labels
   - Categorizes frames into "labeled_frames" and "unlabeled_frames"

3. **Formatting**: The dataset is formatted with various options:

   .. code-block:: python

      formatted_dataset = db.format_dataset(
          dataset=dataset,
          pool_frame_data_by_clip=False,
          include_unlabeled_data=True,
          include_angles=True,
          include_distances=True,
          include_normalized=True,
          segmentation_strategy="flatten_into_columns",
          segmentation_splitter_label="step_type",
          segmentation_window=10,
          segmentation_window_label="weight_transfer_type"
      )

Segmentation Strategies
---------------------

StreamPoseML offers several strategies for segmenting and structuring data:

- **"none"**: Raw frame-by-frame data with no segmentation
- **"split_on_label"**: Segments based on label boundaries
- **"window"**: Fixed-size time windows
- **"flatten_into_columns"**: Temporal data represented as separate columns
- **"flatten_on_example"**: Window-based approach focusing on complete examples

These strategies allow for different ways of representing temporal aspects of pose data.

Example: Complete Annotation Workflow
-----------------------------------

1. **Create annotation files** for your videos with appropriate labels
2. **Process videos** to extract pose keypoints
3. **Build a dataset** by merging annotations with pose data
4. **Format the dataset** with appropriate segmentation and features
5. **Train a model** using the formatted dataset

Below is a complete workflow example based on the actual code in the ``example_usage.ipynb`` notebook:

.. code-block:: python

   # Step 1: Set up input and output directories
   import os
   import time
   
   # Inputs
   example_input_directory = "../../example_data/input"
   example_output_directory = f"../../example_data/output-{time.time_ns()}"
   
   source_annotations_directory = os.path.join(example_input_directory, "source_annotations")
   source_videos_directory = os.path.join(example_input_directory, "source_videos")
   
   # Outputs
   sequence_data_directory = os.path.join(example_output_directory, "sequences")
   keypoints_data_directory = os.path.join(example_output_directory, "keypoints")
   merged_annotation_output_directory = os.path.join(example_output_directory, "datasets")
   trained_models_output_directory = os.path.join(example_output_directory, "trained_models")
   
   for directory in [sequence_data_directory, keypoints_data_directory, 
                     merged_annotation_output_directory, trained_models_output_directory]:
       os.makedirs(directory, exist_ok=True)
   
   # Step 2: Process videos to extract poses
   import stream_pose_ml.jobs.process_videos_job as pv
   
   folder = f"run-preproccessed-{time.time_ns()}"  
   keypoints_path = f"{keypoints_data_directory}/{folder}"
   sequence_path = f"{sequence_data_directory}/{folder}"
   
   data = pv.ProcessVideosJob().process_videos(
       src_videos_path=source_videos_directory,
       output_keypoints_data_path=keypoints_path,
       output_sequence_data_path=sequence_path,
       write_keypoints_to_file=True,
       write_serialized_sequence_to_file=True,
       limit=None,
       configuration={},
       preprocess_video=True,
       return_output=False
   )
   
   # Step 3: Build dataset from annotations and sequences
   import stream_pose_ml.jobs.build_and_format_dataset_job as data_builder
   
   db = data_builder.BuildAndFormatDatasetJob()
   dataset_file_name = "preprocessed_flatten_on_example_10_frames"
   
   dataset = db.build_dataset_from_data_files(
       annotations_data_directory=source_annotations_directory,
       sequence_data_directory=sequence_data_directory,
       limit=None,
   )
   
   # Step 4: Format dataset with desired features
   formatted_dataset = db.format_dataset(
       dataset=dataset,
       pool_frame_data_by_clip=False,
       decimal_precision=4,
       include_unlabeled_data=True,
       include_angles=True,
       include_distances=True,
       include_normalized=True,
       segmentation_strategy="flatten_into_columns",
       segmentation_splitter_label="step_type",
       segmentation_window=10,
       segmentation_window_label="weight_transfer_type",
   )
   
   # Step 5: Write dataset to CSV
   db.write_dataset_to_csv(
       csv_location=merged_annotation_output_directory,
       formatted_dataset=formatted_dataset,
       filename=dataset_file_name
   )
   
   # Step 6: Train a model using the dataset
   from stream_pose_ml.learning import model_builder as mb
   
   # Mapping string categories to numerical values
   value_map = {
       "weight_transfer_type": {
           "Failure Weight Transfer": 0,
           "Successful Weight Transfer": 1,
       },
       "step_type": {
           "Left Step": 0,
           "Right Step": 1,
       },
   }
   # Columns to drop from training
   drop_list = ["video_id", "step_frame_id", "frame_number", "step_type"]
   
   model_builder = mb.ModelBuilder()
   
   model_builder.load_and_prep_dataset_from_csv(
       path=os.path.join(merged_annotation_output_directory, f"{dataset_file_name}.csv"),
       target="weight_transfer_type",
       value_map=value_map,
       column_whitelist=[],
       drop_list=drop_list,
   )
   
   model_builder.set_train_test_split(
       balance_off_target=True,
       upsample_minority=True,
       downsample_majority=False,
       use_SMOTE=False,
       random_state=40002,
   )
   
   # Train a gradient boost model
   model_builder.train_gradient_boost()
   model_builder.evaluate_model()
   
   # Save the model for use in the Web Application
   notes = """Gradient Boost classifier trained on dataset with 10 frame window"""
   model_builder.save_model_and_datasets(
       notes=notes, 
       model_type="gradient-boost", 
       model_path=trained_models_output_directory
   )

Tips for Effective Annotations
----------------------------

1. **Consistency**: Use consistent labels across all videos
2. **Precision**: Ensure accurate frame boundaries for actions
3. **Coverage**: Annotate a diverse range of examples
4. **Balance**: Try to include balanced examples of each class
5. **Verification**: Double-check annotations for accuracy before training

Annotations are the foundation of supervised learning in StreamPoseML. Well-prepared annotations lead to more accurate and effective pose classification models.
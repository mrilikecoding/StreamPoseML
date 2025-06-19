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

.. code-block:: python

   # Step 1: Process videos to extract poses
   import stream_pose_ml.jobs.process_videos_job as pv
   job = pv.ProcessVideosJob()
   job.process_videos(
       src_videos_path='videos/',
       output_keypoints_data_path='keypoints/',
       output_sequence_data_path='sequences/'
   )

   # Step 2: Build dataset from annotations and sequences
   import stream_pose_ml.jobs.build_and_format_dataset_job as data_builder
   db = data_builder.BuildAndFormatDatasetJob()
   
   dataset = db.build_dataset_from_data_files(
       annotations_data_directory='annotations/',
       sequence_data_directory='sequences/'
   )

   # Step 3: Format dataset with desired features
   formatted_dataset = db.format_dataset(
       dataset=dataset,
       include_angles=True,
       include_distances=True,
       segmentation_strategy="flatten_on_example",
       segmentation_window=10
   )

   # Step 4: Write dataset to CSV
   db.write_dataset_to_csv(
       csv_location='datasets/',
       formatted_dataset=formatted_dataset,
       filename="my_training_dataset"
   )

Tips for Effective Annotations
----------------------------

1. **Consistency**: Use consistent labels across all videos
2. **Precision**: Ensure accurate frame boundaries for actions
3. **Coverage**: Annotate a diverse range of examples
4. **Balance**: Try to include balanced examples of each class
5. **Verification**: Double-check annotations for accuracy before training

Annotations are the foundation of supervised learning in StreamPoseML. Well-prepared annotations lead to more accurate and effective pose classification models.
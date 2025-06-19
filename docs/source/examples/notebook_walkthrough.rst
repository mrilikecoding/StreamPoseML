Complete Example Walkthrough
========================

This walkthrough demonstrates a complete end-to-end workflow for StreamPoseML, from video processing to model training.

Step 1: Import StreamPoseML
--------------------------

First, install the package if you haven't already:

.. code-block:: python

   %pip install stream_pose_ml

Step 2: Set Input and Output Directories
---------------------------------------

Set up the directory structure for inputs and outputs:

.. code-block:: python

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

   for dir in [sequence_data_directory, keypoints_data_directory, 
               merged_annotation_output_directory, trained_models_output_directory]:
       os.makedirs(dir, exist_ok=True)

Step 3: Generate Keypoints and Sequence Data
------------------------------------------

Process videos to extract pose keypoints:

.. code-block:: python

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

   print(f"Generated keypoints are located at {data['keypoints_path']}")
   print(f"Generated sequences are located at {data['sequence_path']}")

Step 4: Merge Video Sequence Data into a Dataset
----------------------------------------------

Combine pose data with annotations to create a machine learning dataset:

.. code-block:: python

   import stream_pose_ml.jobs.build_and_format_dataset_job as data_builder 

   db = data_builder.BuildAndFormatDatasetJob()

   dataset_file_name = "preprocessed_flatten_on_example_10_frames_5"

   dataset = db.build_dataset_from_data_files(
       annotations_data_directory=source_annotations_directory,
       sequence_data_directory=sequence_data_directory,
       limit=None,
   )

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

   db.write_dataset_to_csv(
       csv_location=merged_annotation_output_directory,
       formatted_dataset=formatted_dataset,
       filename=dataset_file_name
   )

Step 5: Prepare Training Data
---------------------------

For demonstration purposes, expand the dataset:

.. code-block:: python

   # For demo only: expand a small dataset to create more training data
   import pandas as pd
   data_file = os.path.join(merged_annotation_output_directory, f"{dataset_file_name}.csv")
   data_file_expanded = os.path.join(merged_annotation_output_directory, 
                                     f"{dataset_file_name}-EXPANDED.csv")

   df = pd.read_csv(data_file)
   df = pd.concat(map(pd.read_csv, [data_file for _ in range(100)]), ignore_index=True) 
   df = df.sample(frac=1).reset_index(drop=True)
   df.to_csv(data_file_expanded)

Step 6: Model Training Approaches
---------------------------

### MLflow Integration (Recommended)

For production use, StreamPoseML recommends using MLflow to create model artifacts that can be easily deployed in the web application. The datasets created by StreamPoseML can be used to train models that are then logged with MLflow.

The web application expects a complete MLflow model artifact, which is a gzipped package containing the model and its metadata. To create such artifacts, refer to the `MLflow documentation <https://mlflow.org/docs/latest/ml/model/models-from-code/#logging-the-model>`_ for proper model logging.

Key requirements for creating MLflow models compatible with StreamPoseML:

- Include a sample input example that matches the expected feature format
- Specify required dependencies
- Ensure the model implements a predict() method

### Train a Gradient Boosting Model

Train a model using the prepared dataset:

.. code-block:: python

   from stream_pose_ml.learning import model_builder as mb

   # Mapping string categories to numerical
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
   
   # Columns to drop
   drop_list = ["video_id", "step_frame_id", "frame_number", "step_type"]
   
   # Only keep these columns (empty = keep all except drop_list)
   column_whitelist = []

   model_builder = mb.ModelBuilder()

   model_builder.load_and_prep_dataset_from_csv(
       path=data_file_expanded,
       target="weight_transfer_type",
       value_map=value_map,
       column_whitelist=column_whitelist,
       drop_list=drop_list,
   )

   model_builder.set_train_test_split(
       balance_off_target=True,
       upsample_minority=True,
       downsample_majority=False,
       use_SMOTE=False,
       random_state=40002,
   )
   
   model_builder.train_gradient_boost()
   model_builder.evaluate_model()

Step 7: Train a Random Forest Model (Alternative)
----------------------------------------------

You can also train a Random Forest model with parameter optimization:

.. code-block:: python

   from random import randint
   
   # Use the same dataset preparation as in the Gradient Boost example
   
   model_builder.load_and_prep_dataset_from_csv(
       path=data_file_expanded,
       target="weight_transfer_type",
       value_map=value_map,
       column_whitelist=column_whitelist,
       drop_list=drop_list,
   )

   model_builder.set_train_test_split(
       balance_off_target=True,
       upsample_minority=True,
       downsample_majority=False,
       use_SMOTE=False,
       random_state=40002,
   )

   # Parameter distributions for random search
   param_dist = {
       "n_estimators": [randint(400, 600)],
       "max_depth": [randint(9, 15)],
       "max_features": [randint(4, 12)],
   }
   
   # Fixed parameters
   rf_params = {
       "n_estimators": [20, 50, 100, 200],
       "max_depth": 9,
       "max_leaf_nodes": 63,
   }

   model_builder.train_random_forest(
       use_random_search=True, 
       params=rf_params, 
       param_dist=param_dist, 
       iterations=50, 
       random_state=123
   )
   
   model_builder.evaluate_model()

Step 8: Save Model for Web Application Deployment
----------------------------------------------

StreamPoseML's web application supports two types of model formats:

### Option 1: MLflow Model Artifacts (Recommended)

The web application is designed to work with MLflow model artifacts, which provide better compatibility and deployment features. After logging your model with MLflow, export it as a gzipped model artifact that can be uploaded directly in the web application.

### Option 2: Pickle Files

Alternatively, you can save models using StreamPoseML's built-in method:

.. code-block:: python

   notes = """
   Gradient Boost classifier (90% ROC AUC) trained on dataset preprocessed_flatten_on_example_10_frames, 
   a 10 frame window with flat column 2d angles + distances, and randomly upsampled
   """

   model_builder.save_model_and_datasets(
       notes=notes, 
       model_type="gradient-boost", 
       model_path=trained_models_output_directory
   )

Alternative Dataset Formats
-------------------------

StreamPoseML provides flexibility to create different types of datasets from the same pose sequence data.

Raw Joint Data (One Frame Per Row)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   db = data_builder.BuildAndFormatDatasetJob()
   dataset = db.build_dataset_from_data_files(
       annotations_data_directory=source_annotations_directory,
       sequence_data_directory=sequence_data_directory,
       limit=None,
   )

   formatted_dataset = db.format_dataset(
       dataset=dataset,
       pool_frame_data_by_clip=False,
       decimal_precision=4,
       include_unlabeled_data=True,
       include_joints=True,
       include_z_axis=True,
       include_angles=False,
       include_distances=False,
       include_normalized=False,
       segmentation_strategy="none",
   )

   db.write_dataset_to_csv(
       csv_location=merged_annotation_output_directory,
       formatted_dataset=formatted_dataset,
       filename="preprocessed_frame_joint_data"
   )

Split on Step Type with Pooled Temporal Dynamics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   db = data_builder.BuildAndFormatDatasetJob()
   dataset = db.build_dataset_from_data_files(
       annotations_data_directory=source_annotations_directory,
       sequence_data_directory=sequence_data_directory,
       limit=None,
   )

   formatted_dataset = db.format_dataset(
       dataset=dataset,
       pool_frame_data_by_clip=True,
       decimal_precision=4,
       include_unlabeled_data=True,
       include_angles=True,
       include_distances=True,
       include_normalized=True,
       segmentation_strategy="split_on_label",
       segmentation_splitter_label="step_type",
       segmentation_window=10,
       segmentation_window_label="weight_transfer_type",
   )

   db.write_dataset_to_csv(
       csv_location=merged_annotation_output_directory,
       formatted_dataset=formatted_dataset,
       filename="pooled_angles_distances_last_10_frames"
   )

Flatten Columns Over 10 Frame Window
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   db = data_builder.BuildAndFormatDatasetJob()
   dataset = db.build_dataset_from_data_files(
       annotations_data_directory=source_annotations_directory,
       sequence_data_directory=sequence_data_directory,
       limit=None,
   )

   formatted_dataset = db.format_dataset(
       dataset=dataset,
       pool_frame_data_by_clip=False,
       decimal_precision=4,
       include_unlabeled_data=True,
       include_angles=True,
       include_distances=True,
       include_normalized=True,
       segmentation_strategy="flatten_on_example",
       segmentation_splitter_label="step_type",
       segmentation_window=10,
       segmentation_window_label="weight_transfer_type",
   )

   db.write_dataset_to_csv(
       csv_location=merged_annotation_output_directory,
       formatted_dataset=formatted_dataset,
       filename="flatten_on_example_10_frames_2"
   )

Using Models in the Web Application
--------------------------------

After training and saving your model, deploy it in the StreamPoseML web application:

1. **Start the Web Application**

   .. code-block:: bash

      # From the StreamPoseML root directory
      make start

2. **Access the Web Interface**

   Open http://localhost:3000 in your browser

3. **Upload Your Model**

   - In the Settings section, click the file input field
   - Browse for your saved model file (MLflow gzipped artifact or pickle file)
   - Set the Frame Window Size to match your model's training window size
   - Click "Set Model"

4. **Start Real-time Classification**

   - Click "Classify from webcam stream" to start the camera
   - The system will now process the webcam feed and perform real-time classification
   - View the classification results in the results panel
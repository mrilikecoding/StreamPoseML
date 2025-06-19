End-to-End Tutorial: Dance Movement Classification
==============================================

In this comprehensive tutorial, we'll walk through the complete process of building a dance movement classification system using StreamPoseML. We'll use the example data provided with the package to create a model that can distinguish between successful and unsuccessful weight transfers in dance steps.

What We'll Build
--------------

By the end of this tutorial, you'll have:

1. Processed a dance video to extract pose keypoints
2. Created a labeled dataset combining pose data with movement annotations
3. Trained a model to classify successful vs. unsuccessful weight transfers
4. Set up real-time classification that can analyze new videos

This workflow mirrors real applications in dance education, physical therapy, sports coaching, and other fields where movement quality assessment is important.

.. note::
   This tutorial uses example data included with StreamPoseML. The concepts can be applied to any movement classification task by substituting your own videos and annotations.

Prerequisites
-----------

Before starting, make sure you have:

* StreamPoseML installed: ``pip install stream_pose_ml``
* Example data (included in the package repository)
* Basic understanding of Python and machine learning concepts

Part 1: Project Setup
-------------------

Let's start by setting up our project structure and importing the necessary modules:

.. code-block:: python

   import os
   from pathlib import Path
   import time
   import pandas as pd
   import matplotlib.pyplot as plt
   import cv2
   
   # Import StreamPoseML modules
   import stream_pose_ml
   import stream_pose_ml.jobs.process_videos_job as pv
   import stream_pose_ml.jobs.build_and_format_dataset_job as data_builder
   from stream_pose_ml.learning import model_builder as mb
   
   # Set up project directories using pathlib
   project_dir = Path.cwd() / "dance_classification_project"
   project_dir.mkdir(exist_ok=True, parents=True)
   
   # Setup subdirectories
   data_dir = project_dir / "data"
   input_dir = data_dir / "input"
   output_dir = data_dir / "output"
   
   # Create directories for input data
   videos_dir = input_dir / "videos"
   annotations_dir = input_dir / "annotations"
   
   # Create directories for output data
   keypoints_dir = output_dir / "keypoints"
   sequences_dir = output_dir / "sequences"
   datasets_dir = output_dir / "datasets"
   models_dir = output_dir / "models"
   
   # Create all directories
   for directory in [videos_dir, annotations_dir, keypoints_dir, 
                     sequences_dir, datasets_dir, models_dir]:
       directory.mkdir(exist_ok=True, parents=True)
   
   print(f"Project directory structure created at: {project_dir}")

Now, let's copy the example data into our project:

.. code-block:: python

   import shutil
   
   # Path to example data (adjust if needed)
   # This assumes you're in the StreamPoseML repository root
   example_data_path = Path("example_data")
   
   # Copy example video
   shutil.copy(
       example_data_path / "input" / "source_videos" / "example_video.webm",
       videos_dir / "example_video.webm"
   )
   
   # Copy annotations
   shutil.copy(
       example_data_path / "input" / "source_annotations" / "example_video.json",
       annotations_dir / "example_video.json"
   )
   
   print("Example data copied to project directory")

Part 2: Video Processing
----------------------

Now, let's process our example video to extract pose keypoints:

.. code-block:: python

   # Create a unique folder name for this run
   timestamp = int(time.time())
   run_folder = f"run-{timestamp}"
   
   # Define output paths
   run_keypoints_dir = keypoints_dir / run_folder
   run_sequences_dir = sequences_dir / run_folder
   
   # Process the video to extract pose keypoints
   process_results = pv.ProcessVideosJob().process_videos(
       src_videos_path=str(videos_dir),                  # Where to find the input videos
       output_keypoints_data_path=str(run_keypoints_dir),  # Where to save frame keypoints
       output_sequence_data_path=str(run_sequences_dir),   # Where to save sequences
       write_keypoints_to_file=True,                     # Save individual frame data
       write_serialized_sequence_to_file=True,           # Save continuous sequences
       limit=None,                                       # Process all videos found
       configuration={},                                 # Default configuration
       preprocess_video=True,                            # Apply preprocessing
       return_output=True                                # Return results dictionary
   )
   
   print(f"Processed {len(process_results.get('processed_videos', []))} videos")
   print(f"Keypoints saved to: {run_keypoints_dir}")
   print(f"Sequences saved to: {run_sequences_dir}")

Let's examine what was created:

.. code-block:: python

   # List the keypoint files
   keypoint_files = list(run_keypoints_dir.glob("*.json"))
   
   # Print info about the keypoints
   print(f"Generated {len(keypoint_files)} keypoint files")
   
   # If available, let's look at the sequence files
   sequence_files = list(run_sequences_dir.glob("*.json"))
   print(f"Generated {len(sequence_files)} sequence files")
   
   # Print the first few keypoint files to understand naming
   print("Sample keypoint files:")
   for file in keypoint_files[:5]:
       print(f"  - {file.name}")

Part 3: Creating a Dataset
------------------------

Now, let's combine our extracted pose keypoints with the annotations to create a machine learning dataset:

.. code-block:: python

   # Initialize the dataset builder
   db = data_builder.BuildAndFormatDatasetJob()
   
   # Build the dataset by combining annotations with keypoint sequences
   dataset = db.build_dataset_from_data_files(
       annotations_data_directory=str(annotations_dir),  # Where our annotations are stored
       sequence_data_directory=str(run_sequences_dir),   # Where our sequences are stored
       limit=None,                                      # Process all files
   )
   
   print(f"Created dataset with {len(dataset.all_frames)} total frames")
   print(f"  - Labeled frames: {len(dataset.labeled_frames)}")
   print(f"  - Unlabeled frames: {len(dataset.unlabeled_frames)}")
   
   # Examine the labels in the dataset
   labels = {}
   for frame in dataset.labeled_frames:
       for label in frame.labels:
           labels[label] = labels.get(label, 0) + 1
   
   print("\\nLabel distribution:")
   for label, count in labels.items():
       print(f"  - {label}: {count} frames")

Now, let's format our dataset to include calculated features that will help with classification:

.. code-block:: python

   # Format the dataset with calculated features
   formatted_dataset = db.format_dataset(
       dataset=dataset,
       pool_frame_data_by_clip=False,          # Process frames individually
       decimal_precision=4,                     # Precision for numerical values
       include_unlabeled_data=True,             # Include frames without labels
       include_angles=True,                     # Calculate joint angles
       include_distances=True,                  # Calculate distances between joints
       include_normalized=True,                 # Include normalized coordinates
       include_joints=False,                    # Exclude raw joint positions
       include_z_axis=False,                    # Exclude z-axis data
       segmentation_strategy="flatten_on_example",  # Group frames and flatten features
       segmentation_splitter_label="step_type",     # Split by step type
       segmentation_window=10,                     # Use 10 frame windows
       segmentation_window_label="weight_transfer_type"  # Our target label
   )
   
   # Write the formatted dataset to a CSV file
   dataset_file = f"dance_dataset_{timestamp}.csv"
   db.write_dataset_to_csv(
       csv_location=str(datasets_dir),
       formatted_dataset=formatted_dataset,
       filename=dataset_file.replace('.csv', '')  # Function adds .csv extension
   )
   
   dataset_path = datasets_dir / dataset_file
   print(f"Dataset saved to: {dataset_path}")

Let's examine our dataset:

.. code-block:: python

   # Load the dataset with pandas to explore
   df = pd.read_csv(dataset_path)
   
   print(f"Dataset shape: {df.shape}")
   print(f"Columns: {', '.join(df.columns[:5])}...")
   
   # Check for the target label column
   if 'weight_transfer_type' in df.columns:
       print("\\nWeight transfer type distribution:")
       print(df['weight_transfer_type'].value_counts())
   
   # Look at a few rows
   print("\\nSample data (first 3 rows, first 10 columns):")
   print(df.iloc[:3, :10])

Part 4: Model Training
--------------------

Now that we have a labeled dataset, let's train a model to classify successful vs. unsuccessful weight transfers:

.. code-block:: python

   # Define our label mapping (string labels to numerical values)
   value_map = {
       "weight_transfer_type": {
           "Successful Weight Transfer": 1,
           "Failure Weight Transfer": 0,
       },
       "step_type": {
           "Left Step": 0,
           "Right Step": 1,
       },
   }
   
   # Columns to exclude from training
   drop_list = ["video_id", "step_frame_id", "frame_number", "step_type"]
   
   # Initialize the model builder
   model_builder = mb.ModelBuilder()
   
   # Load and prepare our dataset
   model_builder.load_and_prep_dataset_from_csv(
       path=str(dataset_path),
       target="weight_transfer_type",     # Our classification target
       value_map=value_map,              # Convert text labels to numbers
       column_whitelist=[],              # Use all columns not in drop_list
       drop_list=drop_list,              # Columns to exclude
   )
   
   # Configure train/test split with optional class balancing
   model_builder.set_train_test_split(
       test_size=0.2,                    # Use 20% for testing
       balance_off_target=True,          # Balance classes
       upsample_minority=True,           # Duplicate minority class samples
       downsample_majority=False,        # Don't reduce majority class
       use_SMOTE=False,                  # Don't use SMOTE
       random_state=42,                  # Set random seed for reproducibility
   )
   
   # Train a gradient boosting model (typically good performance for this type of data)
   model_builder.train_gradient_boost(
       params={                          # Hyperparameters
           "n_estimators": 100,          # Number of boosting stages
           "max_depth": 3,               # Max depth of individual trees
           "learning_rate": 0.1,         # Learning rate
       }
   )
   
   # Evaluate the model
   evaluation = model_builder.evaluate_model()
   
   print(f"Model accuracy: {evaluation.get('accuracy', 'N/A')}")
   print(f"F1 score: {evaluation.get('f1_score', 'N/A')}")
   
   # Save the model for later use
   model_name = f"dance_classifier_{timestamp}"
   model_builder.save_model_and_datasets(
       notes="Dance movement classification model - weight transfer success",
       model_type="gradient-boost",
       model_path=str(models_dir),
       model_name=model_name
   )
   
   print(f"Model saved to: {models_dir / f'{model_name}.pkl'}")

Let's visualize the model's performance:

.. code-block:: python

   # If matplotlib is available, plot the ROC curve
   try:
       plt.figure(figsize=(8, 6))
       plt.plot(evaluation.get('fpr'), evaluation.get('tpr'), 
                label=f'ROC curve (area = {evaluation.get("roc_auc", 0):.2f})')
       plt.plot([0, 1], [0, 1], 'k--')
       plt.xlim([0.0, 1.0])
       plt.ylim([0.0, 1.05])
       plt.xlabel('False Positive Rate')
       plt.ylabel('True Positive Rate')
       plt.title('Receiver Operating Characteristic (ROC)')
       plt.legend(loc="lower right")
       plt.savefig(str(models_dir / f"{model_name}_roc.png"))
       plt.close()
       print(f"ROC curve saved to: {models_dir / f'{model_name}_roc.png'}")
       
       # Plot feature importance
       if hasattr(model_builder.model, 'feature_importances_'):
           feature_importances = model_builder.model.feature_importances_
           features = model_builder.X_train.columns
           indices = np.argsort(feature_importances)[-10:]  # Top 10 features
           
           plt.figure(figsize=(10, 8))
           plt.title('Feature Importances')
           plt.barh(range(len(indices)), feature_importances[indices], align='center')
           plt.yticks(range(len(indices)), [features[i] for i in indices])
           plt.xlabel('Relative Importance')
           plt.tight_layout()
           plt.savefig(str(models_dir / f"{model_name}_features.png"))
           plt.close()
           print(f"Feature importance plot saved to: {models_dir / f'{model_name}_features.png'}")
   except Exception as e:
       print(f"Couldn't create visualization: {str(e)}")

Part 5: Real-time Classification
-----------------------------

Now that we have a trained model, let's set up a system for real-time classification:

.. code-block:: python

   import pickle
   from stream_pose_ml import StreamPoseClient
   from stream_pose_ml.blaze_pose.mediapipe_client import MediaPipeClient
   from stream_pose_ml.learning.trained_model import TrainedModel
   from stream_pose_ml.learning.sequence_transformer import SequenceTransformer
   
   # Load our saved model
   model_path = models_dir / f"{model_name}.pkl"
   with open(model_path, 'rb') as f:
       trained_classifier = pickle.load(f)
       model_data = pickle.load(f)
       
   # Create a TrainedModel instance
   model = TrainedModel()
   model.set_model(
       model=trained_classifier,
       model_data=model_data,
       notes="Dance movement classification model"
   )
   
   # Initialize components for real-time classification
   mpc = MediaPipeClient()  # For pose detection
   transformer = SequenceTransformer()  # For feature transformation
   
   # Create a StreamPoseClient for real-time classification
   client = StreamPoseClient(
       frame_window=10,  # Match the window size used in training
       mediapipe_client_instance=mpc,
       trained_model=model,
       data_transformer=transformer
   )
   
   print("Real-time classification system initialized")

Let's demonstrate how to use this for real-time classification with a webcam:

.. code-block:: python

   # Real-time classification from webcam (commented out - uncomment to run)
   '''
   # Open webcam (0 is usually the built-in camera)
   cap = cv2.VideoCapture(0)
   
   # Check if camera opened successfully
   if not cap.isOpened():
       print("Error: Could not open webcam")
   else:
       print("Starting real-time classification. Press 'q' to quit.")
       
       while True:
           # Capture frame-by-frame
           ret, frame = cap.read()
           
           if not ret:
               print("Failed to grab frame")
               break
               
           # Flip the frame horizontally for a selfie-view display
           frame = cv2.flip(frame, 1)
           
           # Process the frame with our StreamPoseClient
           client.run_frame_pipeline(frame)
           
           # Get classification result
           if client.current_classification is not None:
               # Determine label based on classification (boolean)
               label = "Successful Transfer" if client.current_classification else "Failed Transfer"
               color = (0, 255, 0) if client.current_classification else (0, 0, 255)
               
               # Display the result on the frame
               cv2.putText(frame, label, (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
           
           # Display the resulting frame
           cv2.imshow('Dance Movement Classification', frame)
           
           # Break the loop when 'q' is pressed
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break
       
       # Release the capture and close windows
       cap.release()
       cv2.destroyAllWindows()
   '''

Alternatively, we can classify a pre-recorded video:

.. code-block:: python

   # Function to process a video file (can be adjusted to process the original example)
   def classify_video(video_path, output_path=None):
       cap = cv2.VideoCapture(str(video_path))
       
       # Get video properties for output video
       width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
       height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
       fps = cap.get(cv2.CAP_PROP_FPS)
       frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
       
       # Set up video writer if output path is provided
       if output_path:
           fourcc = cv2.VideoWriter_fourcc(*'mp4v')
           out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
       
       # Statistics variables
       frames_processed = 0
       successful_frames = 0
       failed_frames = 0
       
       # Process the video
       while cap.isOpened():
           ret, frame = cap.read()
           if not ret:
               break
               
           # Process the frame
           client.run_frame_pipeline(frame)
           frames_processed += 1
           
           # Check if we have a classification
           if client.current_classification is not None:
               # Update statistics
               if client.current_classification:
                   label = "Successful Transfer"
                   color = (0, 255, 0)  # Green
                   successful_frames += 1
               else:
                   label = "Failed Transfer"
                   color = (0, 0, 255)  # Red
                   failed_frames += 1
                   
               # Display result on frame
               cv2.putText(frame, label, (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
           
           # Write frame to output video
           if output_path:
               out.write(frame)
           
           # Show progress
           if frames_processed % 30 == 0:
               print(f"Processed {frames_processed}/{frame_count} frames "
                     f"({(frames_processed/frame_count)*100:.2f}%)")
       
       # Release resources
       cap.release()
       if output_path:
           out.release()
       
       # Return statistics
       return {
           "frames_processed": frames_processed,
           "successful_frames": successful_frames,
           "failed_frames": failed_frames,
           "success_rate": successful_frames / max(successful_frames + failed_frames, 1)
       }
   
   # Example usage (commented out - uncomment to run)
   '''
   # Process the example video
   results = classify_video(
       video_path=videos_dir / "example_video.webm",
       output_path=models_dir / f"classified_video_{timestamp}.mp4"
   )
   
   print(f"Video classification complete!")
   print(f"Processed {results['frames_processed']} frames")
   print(f"Successful weight transfers: {results['successful_frames']} frames")
   print(f"Failed weight transfers: {results['failed_frames']} frames")
   print(f"Overall success rate: {results['success_rate']*100:.2f}%")
   '''

Part 6: Deployment Options
------------------------

Now that we have a working system, let's explore deployment options:

1. **Python Integration**:

   You can integrate the classifier into any Python application:

   .. code-block:: python
   
      from stream_pose_ml import StreamPoseClient
      
      # Initialize components (as shown above)
      
      # In your application's main loop:
      def process_frame(frame):
          client.run_frame_pipeline(frame)
          result = client.current_classification
          return result

2. **Web Application**:

   The StreamPoseML web application provides a ready-to-use interface:

   .. code-block:: bash
   
      # Clone the repository (if you haven't already)
      git clone https://github.com/mrilikecoding/StreamPoseML.git
      cd StreamPoseML
      
      # Start the web application
      make start
      
      # Open http://localhost:3000 in your browser
      # You can upload your trained model in the Settings section

3. **MLflow Integration**:

   For production deployments, use MLflow integration:

   .. code-block:: python
   
      import mlflow
      
      # Log the model with MLflow
      with mlflow.start_run(run_name="dance_classifier"):
          # Log model parameters
          mlflow.log_params({
              "window_size": 10,
              "feature_type": "angles_and_distances",
              "model_type": "gradient_boost"
          })
          
          # Log model metrics
          mlflow.log_metrics({
              "accuracy": evaluation.get("accuracy", 0),
              "f1_score": evaluation.get("f1_score", 0),
              "roc_auc": evaluation.get("roc_auc", 0)
          })
          
          # Log the model itself (works with sklearn models)
          mlflow.sklearn.log_model(
              sk_model=model_builder.model,
              artifact_path="model",
              registered_model_name="dance_movement_classifier"
          )

Conclusion
---------

In this tutorial, we've built a complete dance movement classification system using StreamPoseML. We:

1. Processed a video to extract pose data
2. Created a labeled dataset with meaningful features
3. Trained a machine learning model
4. Set up real-time classification
5. Explored deployment options

This same workflow can be applied to many movement analysis tasks:

- Sports technique assessment
- Physical therapy monitoring
- Dance education
- Fitness form correction
- Ergonomic movement analysis

By adjusting the input videos, annotations, and features, you can adapt this approach to your specific movement classification needs.
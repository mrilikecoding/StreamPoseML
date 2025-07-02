Quick Start Guide
=================

Welcome to StreamPoseML! This guide will help you get up and running in just a few minutes. This guide shows you how to process videos and classify movements.

In this guide, you'll learn how to:

1. **Install** StreamPoseML
2. **Process** a video to extract movement data
3. **Create** a dataset for machine learning
4. **Train** a simple pose classification model
5. **Deploy** your model for real-time classification

Installation
-----------

Let's start by installing StreamPoseML and its dependencies:

.. code-block:: bash

   # Install using pip
   pip install stream-pose-ml
   
   # Verify installation
   python -c "import stream_pose_ml; print(f'StreamPoseML version: {stream_pose_ml.__version__}')"

Basic Usage
-----------

StreamPoseML provides tools for analyzing human movement in videos. Let's import the modules we'll need:

.. code-block:: python

   # Import key modules
   import stream_pose_ml
   from pathlib import Path  # For cleaner path handling

   # Import job modules for video processing and dataset creation
   import stream_pose_ml.jobs.process_videos_job as pv
   import stream_pose_ml.jobs.build_and_format_dataset_job as data_builder

Video Processing
----------------

Let's extract pose keypoints from videos. First, we'll set up our paths using ``pathlib`` for cleaner path handling:

.. code-block:: python

   # Set up our project directories
   project_dir = Path.cwd() / "my_movement_project"
   
   # Input video location
   videos_dir = project_dir / "videos"
   
   # Output directories for processed data
   keypoints_dir = project_dir / "keypoints"
   sequences_dir = project_dir / "sequences"
   
   # Create directories if they don't exist
   keypoints_dir.mkdir(exist_ok=True, parents=True)
   sequences_dir.mkdir(exist_ok=True, parents=True)

Now let's process our videos to extract pose information:

.. code-block:: python

   # Process videos and extract pose keypoints
   results = pv.ProcessVideosJob().process_videos(
       src_videos_path=str(videos_dir),             # Directory with your input videos
       output_keypoints_data_path=str(keypoints_dir),  # Where to save frame-by-frame keypoints
       output_sequence_data_path=str(sequences_dir),   # Where to save sequence data
       write_keypoints_to_file=True,                # Save individual frame data
       write_serialized_sequence_to_file=True       # Save continuous sequences
   )
   
   print(f"Processed {len(results.get('processed_videos', []))} videos")

This extracts body keypoints from each frame in your videos and saves them in two formats:
1. Individual frames with keypoint data
2. Continuous sequences that combine multiple frames

Creating a Dataset
------------------

Next, we'll combine our keypoint data with movement labels to create a machine learning dataset. 

For this example, imagine we have annotation files that label specific movements in our videos (like "successful dance step" or "incorrect form").

.. code-block:: python

   # Set up paths for annotations and output dataset
   annotations_dir = project_dir / "annotations"
   dataset_output_dir = project_dir / "datasets"
   dataset_output_dir.mkdir(exist_ok=True, parents=True)
   
   # Initialize the dataset builder
   db = data_builder.BuildAndFormatDatasetJob()

   # Build dataset by combining annotations with extracted pose sequences
   # The annotations tell us which frames contain which movements/labels
   dataset = db.build_dataset_from_data_files(
       annotations_data_directory=str(annotations_dir),  # Where your annotation files are stored
       sequence_data_directory=str(sequences_dir)       # The sequences we created in the previous step
   )
   
   print(f"Created dataset with {len(dataset.all_frames)} frames, {len(dataset.labeled_frames)} labeled")

   # Format the dataset with calculated features to improve classification
   formatted_dataset = db.format_dataset(
       dataset=dataset,
       include_angles=True,      # Calculate joint angles (like elbow bend)
       include_distances=True,   # Calculate distances between joints
       include_normalized=True,  # Use normalized coordinates for size independence
       segmentation_strategy="none"  # Process each frame individually
   )

   # Save the formatted dataset as a CSV file for training
   db.write_dataset_to_csv(
       csv_location=str(dataset_output_dir),
       formatted_dataset=formatted_dataset,
       filename="movement_dataset"  # Will create "movement_dataset.csv"
   )
   
   print(f"Saved dataset to {dataset_output_dir / 'movement_dataset.csv'}")
   
   # Quick peek at our data
   import pandas as pd
   df = pd.read_csv(dataset_output_dir / 'movement_dataset.csv')
   print(f"Dataset shape: {df.shape}, columns: {', '.join(df.columns[:5])}...")
   print(f"Labels: {df['label'].value_counts().to_dict() if 'label' in df.columns else 'No label column'}")
   
   # Optional: Examine the dataset structure
   # print(formatted_dataset.keys())
   # print(formatted_dataset['rows'][:2])  # First two rows

Training a Model
--------------

Now let's train a simple classification model using our dataset:

.. code-block:: python

   from stream_pose_ml.learning import model_builder as mb
   
   # Set up a path to save our trained model
   models_dir = project_dir / "models"
   models_dir.mkdir(exist_ok=True, parents=True)
   
   # Create our model builder
   model_builder = mb.ModelBuilder()
   
   # If your target column has string labels (like "correct" and "incorrect"),
   # create a mapping to convert them to numbers
   value_map = {
       "movement_type": {  # Replace with your actual label column name
           "correct_form": 1,
           "incorrect_form": 0,
       }
   }
   
   # Load our dataset
   model_builder.load_and_prep_dataset_from_csv(
       path=str(dataset_output_dir / "movement_dataset.csv"), 
       target="movement_type",  # The column containing your labels
       value_map=value_map,     # Convert text labels to numbers
       column_whitelist=[],     # Empty = use all non-dropped columns
       drop_list=["frame_id", "video_id"]  # Columns to exclude
   )
   
   # Configure train/test split with optional class balancing
   model_builder.set_train_test_split(
       test_size=0.2,           # Use 20% of data for testing
       balance_off_target=True, # Ensure classes are balanced
       upsample_minority=True,  # Duplicate samples from minority class
       random_state=42          # Set seed for reproducibility
   )
   
   # Train a gradient boosting model
   model_builder.train_gradient_boost()
   
   # Evaluate the model's performance
   evaluation = model_builder.evaluate_model()
   print(f"Model accuracy: {evaluation.get('accuracy', 'N/A')}")
   
   # Save the trained model
   model_builder.save_model_and_datasets(
       notes="Movement classification model trained on example dataset",
       model_type="gradient-boost",
       model_path=str(models_dir)
   )
   
   print(f"Model saved to {models_dir}")

Real-time Classification
------------------------

Finally, let's use our trained model for real-time classification:

.. code-block:: python

   from stream_pose_ml import StreamPoseClient
   from stream_pose_ml.blaze_pose.mediapipe_client import MediaPipeClient
   from stream_pose_ml.learning.trained_model import TrainedModel
   from stream_pose_ml.learning.sequence_transformer import SequenceTransformer
   import pickle
   import cv2

   # Load our saved model
   model = TrainedModel()
   with open(models_dir / "gradient-boost-model.pkl", 'rb') as f:
       trained_classifier = pickle.load(f)
       model_data = pickle.load(f)
   
   model.set_model(model=trained_classifier, model_data=model_data)

   # Initialize components
   mpc = MediaPipeClient()  # Connects to MediaPipe for pose detection
   transformer = SequenceTransformer()  # Transforms pose data to features

   # Create a StreamPoseClient for real-time classification
   client = StreamPoseClient(
       frame_window=30,  # How many frames to consider together (like 1 second of video)
       mediapipe_client_instance=mpc,
       trained_model=model,
       data_transformer=transformer
   )

   # Example: Process a single image (in a real app, you'd process video frames)
   image_path = Path("path/to/test/image.jpg")
   if image_path.exists():
       image = cv2.imread(str(image_path))
       client.run_frame_pipeline(image)
       
       # Get the classification result
       result = client.current_classification
       print(f"Classification result: {result}")
   else:
       print(f"Image not found at {image_path}")

   # For webcam processing, you would use something like:
   '''
   cap = cv2.VideoCapture(0)  # Open webcam
   while True:
       ret, frame = cap.read()
       if not ret:
           break
           
       # Process the frame
       client.run_frame_pipeline(frame)
       
       # Use the classification
       if client.current_classification is not None:
           # Do something with the result
           label = "Correct form" if client.current_classification else "Incorrect form"
           cv2.putText(frame, label, (50, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
       
       cv2.imshow('Movement Classification', frame)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
           
   cap.release()
   cv2.destroyAllWindows()
   '''

Web Application Deployment
-----------------------

For a complete interactive experience, you can deploy your model in the StreamPoseML web application:

.. code-block:: bash

   # Clone the repository (if you haven't already)
   git clone https://github.com/mrilikecoding/StreamPoseML.git
   cd StreamPoseML
   
   # Start the web application (uses Docker)
   make start
   
   # Now open http://localhost:3000 in your browser
   # You can upload your model in the Settings section

Next Steps
----------

**For Python Package Users:**

* :doc:`../workflows/video_processing` - Detailed video processing workflow
* :doc:`../guide/concepts` - Core concepts and dataset creation
* :doc:`../api/clients` - Package reference for model training and usage
* :doc:`../examples/notebook_walkthrough` - Complete example with real data

**For Web Application Users:**

* :doc:`../webapp/installation` - Setting up the web application
* :doc:`../webapp/usage` - Using the web application
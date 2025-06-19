Client Interfaces
================

StreamPoseML's Integration Options
----------------------------------------

This guide helps you understand and choose between StreamPoseML's client interfaces to integrate pose-based classification into your applications.

Choosing the Right Client Interface
--------------------------------

StreamPoseML offers two main client interfaces, each designed for different use cases:

* **StreamPoseClient**: A lightweight, self-contained client that processes video frames directly and handles the entire pipeline from pose detection to classification. Best for:
  - Desktop applications
  - Standalone Python applications
  - Projects where simplicity is more important than scalability
  - When you need to process raw video frames

* **MLFlowClient**: A client that leverages MLflow for model management. Best for:
  - Web applications
  - Production systems requiring scalability
  - Projects needing model versioning and management
  - When you need standardized model deployment

StreamPoseClient: Simple and Direct
-----------------------------

The `StreamPoseClient` provides a straightforward way to perform real-time pose classification directly from video frames or pre-extracted keypoints. It's perfect for applications where you want to keep everything in Python and need a complete end-to-end solution.

.. figure:: /_static/logo.png
   :align: center
   :alt: StreamPoseClient workflow
   
   *StreamPoseClient handles the entire pipeline from frame capture to classification*

**Key Features:**

- **Complete Pipeline**: Handles all steps from pose detection through classification
- **Temporal Analysis**: Maintains a window of frames for analyzing movements over time
- **Flexible Input**: Works with raw video frames or pre-extracted keypoints
- **Real-time Focus**: Optimized for low-latency classification
- **Self-contained**: No external services or dependencies needed for operation

Basic Usage
~~~~~~~~~

.. code-block:: python

    from stream_pose_ml import StreamPoseClient
    from stream_pose_ml.blaze_pose.mediapipe_client import MediaPipeClient
    from stream_pose_ml.learning.trained_model import TrainedModel
    from stream_pose_ml.learning.sequence_transformer import SequenceTransformer
    import pickle
    
    # 1. Load your trained model
    model = TrainedModel()
    with open('path/to/your/model.pkl', 'rb') as f:
        trained_classifier = pickle.load(f)
        model_data = pickle.load(f)
    
    model.set_model(model=trained_classifier, model_data=model_data)
    
    # 2. Initialize components
    mpc = MediaPipeClient()  # MediaPipe client for pose detection
    transformer = SequenceTransformer()  # Transforms pose data to features
    
    # 3. Create StreamPoseClient
    client = StreamPoseClient(
        frame_window=30,  # Number of frames to consider together (like 1 second of video)
        mediapipe_client_instance=mpc,
        trained_model=model,
        data_transformer=transformer
    )
    
    # 4. Process video frames
    import cv2
    
    # Option A: Process a single image
    image = cv2.imread('path/to/image.jpg')
    client.run_frame_pipeline(image)
    
    # Option B: Process keypoints (if already extracted)
    # client.run_keypoint_pipeline(keypoints_data)
    
    # 5. Get the classification result
    result = client.current_classification
    print(f"Classification result: {result}")
    
    # 6. For continuous video like webcam feed:
    '''
    cap = cv2.VideoCapture(0)  # Open webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process each frame
        client.run_frame_pipeline(frame)
        
        # Use the classification result when available
        if client.current_classification is not None:
            # Do something with the result
            print(f"Classification: {client.current_classification}")
        
        # Exit on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    '''

**Parameters Explained:**

- **frame_window** (int): The number of consecutive frames to analyze together. Important for movements that happen over time. If your video is 30 fps and you want to analyze 1 second of movement, use 30.
- **mediapipe_client_instance**: An instance of MediaPipeClient that handles pose detection.
- **trained_model**: Your previously trained model wrapped in a TrainedModel class.
- **data_transformer**: Transforms raw pose data into the feature format your model expects.

Practical Example: Video Analysis Application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's how to integrate StreamPoseClient into a video analysis application:

.. code-block:: python

    import cv2
    import numpy as np
    from stream_pose_ml import StreamPoseClient
    from stream_pose_ml.blaze_pose.mediapipe_client import MediaPipeClient
    from stream_pose_ml.learning.trained_model import TrainedModel
    
    # Initialize your model and components first (as shown above)
    # ...
    
    # Define a function for processing video files
    def analyze_video(video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        
        # Set up video writer if saving output
        if output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Classification statistics
        frames = 0
        positive_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame
            client.run_frame_pipeline(frame)
            frames += 1
            
            # Visualize classification when available
            if client.current_classification is not None:
                # Type could be "correct form", "incorrect form", etc.
                # depending on what your model predicts
                if client.current_classification:
                    label = "Correct Movement"
                    color = (0, 255, 0)  # Green
                    positive_frames += 1
                else:
                    label = "Incorrect Movement"
                    color = (0, 0, 255)  # Red
                
                # Display on frame
                cv2.putText(frame, label, (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Write frame if saving output
            if output_path:
                out.write(frame)
        
        # Clean up
        cap.release()
        if output_path:
            out.release()
        
        return {
            "total_frames": frames,
            "positive_frames": positive_frames,
            "positive_percentage": (positive_frames / frames * 100) if frames > 0 else 0
        }
    
    # Example usage
    results = analyze_video("dance_video.mp4", "analyzed_video.mp4")
    print(f"Analysis complete: {results['positive_percentage']:.2f}% correct movements")

MLFlowClient: Complex Model Deployment
-------------------------------------

The `MLFlowClient` is designed for systems where model management, versioning, and scalability are critical. It integrates with MLflow for robust model deployment and is particularly well-suited for web applications.

.. figure:: /_static/logo.png
   :align: center
   :alt: MLFlowClient architecture
   
   *MLFlowClient integrates with MLflow for scalable model serving*

**Key Features:**

- **MLflow Integration**: Connects directly to MLflow model serving endpoints
- **Optimized Data Flow**: Designed for web applications receiving keypoint data
- **Frame Overlap**: Supports smoother predictions with overlapping frame windows
- **Performance Tracking**: Monitors prediction times and model performance metrics
- **Scalability**: Works well in distributed architectures and cloud deployments

Basic Usage
~~~~~~~~~

.. code-block:: python

    import requests
    from stream_pose_ml import MLFlowClient
    from stream_pose_ml.blaze_pose.mediapipe_client import MediaPipeClient
    from stream_pose_ml.learning.sequence_transformer import SequenceTransformer
    
    # 1. Initialize dependencies
    mpc = MediaPipeClient(dummy_client=True)  # dummy_client=True if not processing raw frames
    transformer = SequenceTransformer()  # Or specialized MLFlowTransformer if needed
    
    # 2. Define a prediction function that interfaces with MLflow
    def mlflow_predict(json_data_payload):
        # Send request to MLflow serving endpoint
        response = requests.post(
            "http://mlflow:5002/invocations",  # Your MLflow server endpoint
            json={"inputs": json_data_payload}, 
            headers={"Content-Type": "application/json"}
        )
        return response.json()
    
    # 3. Create client with frame overlap for smoother predictions
    client = MLFlowClient(
        mediapipe_client_instance=mpc,
        data_transformer=transformer,
        predict_fn=mlflow_predict,  # Your custom prediction function
        input_example={"columns": ["angle_left_elbow", "angle_right_knee"]},  # Match your model's expected input
        frame_window=30,  # Total frames to consider
        frame_overlap=5   # Process new predictions every (frame_window - frame_overlap) frames
    )
    
    # 4. Process keypoints (typically from a web client)
    keypoints_data = {"joint_positions": {...}}  # Received from frontend or other source
    client.run_keypoint_pipeline(keypoints_data)
    
    # 5. Get classification result
    result = client.current_classification
    processing_time = client.prediction_processing_time  # Performance monitoring
    
    print(f"Classification: {result}, Processing time: {processing_time}ms")

**Parameters Explained:**

- **mediapipe_client_instance**: MediaPipeClient instance (often with dummy_client=True for web deployments)
- **data_transformer**: Transforms pose data into the format your MLflow model expects
- **predict_fn**: Custom function that sends data to MLflow and returns predictions
- **input_example**: Example of the input format your model expects
- **frame_window**: Number of frames to consider in each analysis window
- **frame_overlap**: Number of frames that overlap between consecutive analysis windows

Web Application Integration
-------------------------

Here's a complete example of integrating MLFlowClient in a Flask web application with WebSockets for real-time communication:

.. code-block:: python

    from flask import Flask, request, jsonify
    from flask_socketio import SocketIO, emit
    import requests
    from stream_pose_ml import MLFlowClient
    from stream_pose_ml.blaze_pose.mediapipe_client import MediaPipeClient
    from stream_pose_ml.learning.sequence_transformer import SequenceTransformer
    
    # Initialize Flask and SocketIO
    app = Flask(__name__)
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    # Global client instance
    stream_pose_client = None
    
    # MLflow prediction function
    def mlflow_predict(json_data):
        """Send prediction requests to MLflow serving endpoint"""
        try:
            response = requests.post(
                "http://mlflow:5002/invocations", 
                json={"inputs": json_data}, 
                headers={"Content-Type": "application/json"}
            )
            return response.json()
        except Exception as e:
            app.logger.error(f"Prediction error: {str(e)}")
            return None
    
    # API endpoint to set model parameters
    @app.route("/api/model/setup", methods=["POST"])
    def setup_model():
        """Configure the MLFlowClient with model parameters"""
        global stream_pose_client
        
        data = request.json
        frame_window = data.get("frame_window", 30)
        frame_overlap = data.get("frame_overlap", 5)
        input_example = data.get("input_example", {"columns": []})
        
        try:
            # Initialize components
            mpc = MediaPipeClient(dummy_client=True)  # No raw frame processing
            transformer = SequenceTransformer()
            
            # Create MLFlowClient
            stream_pose_client = MLFlowClient(
                mediapipe_client_instance=mpc,
                data_transformer=transformer,
                predict_fn=mlflow_predict,
                input_example=input_example,
                frame_window=frame_window,
                frame_overlap=frame_overlap,
            )
            
            return jsonify({"status": "success", "message": "Model configured"})
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    # WebSocket endpoint for real-time keypoint processing
    @socketio.on("keypoints")
    def handle_keypoints(payload):
        """Process incoming keypoint data and return classification"""
        global stream_pose_client
        
        if stream_pose_client is None:
            emit("frame_result", {"error": "No model configured"})
            return
        
        try:    
            # Process keypoints with MLFlowClient
            results = stream_pose_client.run_keypoint_pipeline(payload)
            
            # Return classification results if available
            if results and stream_pose_client.current_classification is not None:
                classification = stream_pose_client.current_classification
                predict_speed = stream_pose_client.prediction_processing_time
                
                # Send results back to client
                emit("frame_result", {
                    "classification": classification,
                    "prediction_time": predict_speed,
                    "confidence": stream_pose_client.current_confidence 
                               if hasattr(stream_pose_client, "current_confidence") else None
                })
            else:
                emit("frame_result", {"status": "processing"})
                
        except Exception as e:
            app.logger.error(f"Error processing keypoints: {str(e)}")
            emit("frame_result", {"error": str(e)})
    
    if __name__ == "__main__":
        socketio.run(app, host="0.0.0.0", port=5000, debug=True)

TrainedModel: Your Machine Learning Container
----------------------------------------

The `TrainedModel` class is a convenient container for your trained machine learning models. It provides a standard interface regardless of the underlying model type and handles integration with data transformers.

**Key Features:**

- **Model Encapsulation**: Neatly packages your trained model and associated data
- **Consistent Interface**: Provides a standardized predict() method regardless of model type
- **Data Transformer Integration**: Works seamlessly with StreamPoseML's transformers
- **Metadata Support**: Stores additional information about the model

Basic Usage
~~~~~~~~~

.. code-block:: python

    from stream_pose_ml.learning.trained_model import TrainedModel
    import pickle
    import joblib
    
    # Create a TrainedModel instance
    model = TrainedModel()
    
    # Load a model from pickle file
    with open('dance_classifier.pkl', 'rb') as f:
        trained_classifier = pickle.load(f)  # The actual model (e.g., RandomForest, XGBoost)
        model_data = pickle.load(f)  # Additional data like test features
    
    # Set the model and associated data
    model.set_model(
        model=trained_classifier,  # Your trained sklearn/xgboost/etc. model
        model_data={               # Additional data needed for predictions
            "X_test": model_data["X_test"],  # Feature columns from test data
            "feature_names": model_data.get("feature_names", []),
        },
        notes="Dance movement classifier trained on 500 examples" # Optional documentation
    )
    
    # Optional: Connect a data transformer for preprocessing
    from stream_pose_ml.learning.sequence_transformer import SequenceTransformer
    transformer = SequenceTransformer()
    model.set_data_transformer(transformer)
    
    # Make predictions directly
    import numpy as np
    test_features = np.array([[0.5, 0.3, 0.2, ...]])  # Your feature vector
    predictions = model.predict(data=test_features)
    
    print(f"Prediction: {predictions[0]}")

**When to use TrainedModel:**

- When integrating with StreamPoseClient or MLFlowClient
- To standardize prediction interfaces across different model types
- To package models for deployment in the StreamPoseML ecosystem

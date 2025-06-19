Client Interfaces
================

This page describes the main client interfaces provided by StreamPoseML and how to integrate them with external applications.

Client Interface Selection Guide
-------------------------------

When integrating StreamPoseML with your application, choose the appropriate client interface based on your needs:

* **StreamPoseClient**: Best for applications that directly process video frames and need lightweight, self-contained classification.
* **MLFlowClient**: Ideal for production deployments that require scalable model serving, versioning, and performance monitoring.

StreamPoseClient
--------------

The StreamPoseClient is the primary interface for real-time pose classification directly from video frames or pre-extracted keypoints.

.. code-block:: python

    from stream_pose_ml import StreamPoseClient
    from stream_pose_ml.blaze_pose.mediapipe_client import MediaPipeClient
    from stream_pose_ml.transformers.sequence_transformer import TenFrameFlatColumnAngleTransformer
    
    # Initialize dependencies
    mpc = MediaPipeClient()
    transformer = TenFrameFlatColumnAngleTransformer()
    model = TrainedModel()  # Load your trained model
    
    # Create client with a window of 30 frames
    client = StreamPoseClient(
        frame_window=30,
        mediapipe_client_instance=mpc,
        trained_model=model,
        data_transformer=transformer
    )
    
    # Option 1: Process video frame directly
    client.run_frame_pipeline(image)
    
    # Option 2: Process pre-extracted keypoints
    client.run_keypoint_pipeline(keypoints)
    
    # Get classification result
    result = client.current_classification

**Key Features:**
- Maintains a window of frames for temporal analysis
- Processes raw video frames through MediaPipe
- Can also accept pre-extracted keypoints from JavaScript clients
- Provides real-time classification results

**Integration Example:**

.. code-block:: python

    # Example integration in a video processing application
    import cv2
    from stream_pose_ml import StreamPoseClient
    
    # Initialize client (see above)
    
    # Open video capture
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process the frame
        client.run_frame_pipeline(frame)
        
        # Get and use classification result
        if client.current_classification:
            # Handle positive classification
            cv2.putText(frame, "Detected", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

MLFlowClient
-----------

The MLFlowClient is an interface for integrating with MLflow for scalable model serving, especially for web applications and services.

.. code-block:: python

    from stream_pose_ml import MLFlowClient
    from stream_pose_ml.blaze_pose.mediapipe_client import MediaPipeClient
    from stream_pose_ml.transformers.sequence_transformer import MLFlowTransformer
    
    # Initialize dependencies
    mpc = MediaPipeClient(dummy_client=True)  # dummy_client=True if not processing raw frames
    transformer = MLFlowTransformer()
    
    # Define a prediction function that interfaces with MLflow
    def mlflow_predict(json_data_payload):
        # Send request to MLflow serving endpoint
        response = requests.post(
            "http://mlflow:5002/invocations", 
            json={"inputs": json_data_payload}, 
            headers={"Content-Type": "application/json"}
        )
        return response.json()
    
    # Create client with frame overlap for smoother predictions
    client = MLFlowClient(
        mediapipe_client_instance=mpc,
        data_transformer=transformer,
        predict_fn=mlflow_predict,
        input_example={"columns": ["joint_x", "joint_y", "joint_z"]},
        frame_window=30,
        frame_overlap=5  # Process new predictions every 25 frames
    )
    
    # Process keypoints
    client.run_keypoint_pipeline(keypoints)
    
    # Get classification result
    result = client.current_classification

**Key Features:**
- Integrates with MLflow for model serving
- Supports frame overlap for continuous analysis with less computational overhead
- Optimized for web applications receiving keypoint data
- Tracks prediction performance metrics

**API Service Integration Example:**

The StreamPoseML API service demonstrates how to use MLFlowClient in a Flask web application:

.. code-block:: python

    # Example from the StreamPoseML API service
    from flask import Flask, request
    from flask_socketio import SocketIO, emit
    
    # Initialize Flask and SocketIO
    app = Flask(__name__)
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    # Set up MLflow client (simplified)
    def set_ml_flow_client(input_example, frame_window=30, frame_overlap=5):
        transformer = MLFlowTransformer()
        
        stream_pose_app.set_stream_pose_client(
            MLFlowClient(
                mediapipe_client_instance=mpc,
                data_transformer=transformer,
                predict_fn=mlflow_predict,
                input_example=input_example,
                frame_window=frame_window,
                frame_overlap=frame_overlap,
            )
        )
    
    # Handle incoming keypoint data via WebSockets
    @socketio.on("keypoints")
    def handle_keypoints(payload):
        if stream_pose_client is None:
            emit("frame_result", {"error": "No model set"})
            return
            
        # Process keypoints
        results = stream_pose_client.run_keypoint_pipeline(payload)
        
        # Return classification results
        if results and stream_pose_client.current_classification is not None:
            classification = stream_pose_client.current_classification
            predict_speed = stream_pose_client.prediction_processing_time
            
            emit("frame_result", {
                "classification": classification,
                "prediction_time": predict_speed
            })

TrainedModel
-----------

The TrainedModel class encapsulates a trained machine learning model and handles integration with data transformers.

.. code-block:: python

    from stream_pose_ml.learning.trained_model import TrainedModel
    import pickle
    
    # Create a TrainedModel instance
    model = TrainedModel()
    
    # Option 1: Load a saved model
    with open('model.pkl', 'rb') as f:
        trained_classifier = pickle.load(f)
    
    # Set the model and associated data
    model.set_model(
        model=trained_classifier,
        model_data={"X_test": test_data},
        notes="Model trained on pose data"
    )
    
    # Connect a data transformer for preprocessing
    model.set_data_transformer(transformer)
    
    # Option 2: Make predictions directly
    predictions = model.predict(data=transformed_data)

**Key Features:**
- Stores both the model and its associated test data
- Links with the appropriate data transformer
- Provides a consistent prediction interface regardless of underlying model type
- Supports additional metadata about the model
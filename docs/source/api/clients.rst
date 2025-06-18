Client Interfaces
================

This page describes the main client interfaces provided by StreamPoseML.

StreamPoseClient
--------------

The StreamPoseClient is the primary interface for real-time pose classification.

.. code-block:: python

    from stream_pose_ml import StreamPoseClient
    
    # Create client
    client = StreamPoseClient(
        frame_window=30,
        mediapipe_client_instance=mpc,
        trained_model=model,
        data_transformer=transformer
    )
    
    # Process video frame
    client.run_frame_pipeline(image)
    
    # Get classification result
    result = client.current_classification

MLFlowClient
-----------

The MLFlowClient is an interface for integrating with MLflow for model serving.

.. code-block:: python

    from stream_pose_ml import MLFlowClient
    
    # Create client
    client = MLFlowClient(
        mediapipe_client_instance=mpc,
        predict_fn=mlflow_predict,
        input_example={"columns": ["joint_x", "joint_y", "joint_z"]},
        frame_window=30,
        frame_overlap=5
    )
    
    # Process keypoints
    client.run_keypoint_pipeline(keypoints)
    
    # Get classification result
    result = client.current_classification

TrainedModel
-----------

The TrainedModel class encapsulates a trained machine learning model.

.. code-block:: python

    from stream_pose_ml.learning.trained_model import TrainedModel
    
    # Create a TrainedModel instance
    model = TrainedModel()
    
    # Set a trained model
    model.set_model(
        model=trained_classifier,
        model_data={"X_test": test_data},
        notes="Model trained on pose data"
    )
    
    # Make predictions
    predictions = model.predict(test_data)
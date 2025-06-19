Usage
=====

Accessing the Web Application
----------------------------

After starting the application with ``make start``, the web interface will automatically open in your default browser. If it doesn't, you can access it at:

http://localhost:3000

The API is available at:

http://localhost:5001

Web Interface Components
-----------------------

The StreamPoseML web interface consists of several main components:

1. **Model Configuration Panel** - For selecting and configuring models
2. **Classification Results** - Displays real-time classification output
3. **Camera Control** - For starting/stopping webcam capture
4. **Visualization Area** - Shows video feed with pose overlay
5. **Bluetooth Integration** - For connecting to external devices

Model Configuration
------------------

### Loading a Model

StreamPoseML supports two ways to load models: direct pickle files and MLflow-logged models.

### Option 1: Using Pickle Files

1. **Prepare your model**:
   
   Your trained model should be saved as a pickle file with a structure like:

   .. code-block:: python

      {
        "classifier": <your_trained_model implementing predict method>
      }

   The model should implement a ``predict`` method that takes an array of examples to classify.

2. **Select the model in the UI**:

   - Open the web UI at http://localhost:3000
   - Go to Settings
   - Click the file input to browse and select your model file from anywhere on your system
   - Configure the Frame Window Size and Prediction Frame Overlap as needed
   - Click "Set Model"

### Option 2: Using MLflow Models

StreamPoseML has built-in support for models logged with MLflow (compatible with MLflow versions < 2.21):

1. **Log your model with MLflow**:

   .. code-block:: python

      import mlflow
      import numpy as np
      
      # Start an MLflow run
      with mlflow.start_run():
          # Log model parameters
          mlflow.log_params(params)
          
          # Log model metrics
          mlflow.log_metrics(metrics)
          
          # Create sample input for model signature
          # This is important for automatic serving
          sample_input = np.array([[...]])  # Your feature array shape
          
          # Log the model with input example
          mlflow.sklearn.log_model(
              model, 
              "model",
              input_example=sample_input
          )
   
   For more details on model logging with signatures and input examples, see the
   `MLflow Models documentation <https://mlflow.org/docs/latest/models.html#model-signature-and-input-example>`_.

2. **Configure the MLflow connection**:

   The StreamPoseML web application includes a dedicated MLflow container that connects to your MLflow tracking server. By default, it looks for models in the local MLflow server instance running at `http://localhost:5002`.

3. **Select the MLflow model in the UI**:

   - Open the web UI at http://localhost:3000
   - Go to Settings
   - Upload your MLflow model file (gzip format with all needed assets)
   - Configure the Frame Window Size and Prediction Frame Overlap as needed
   - Click "Set Model"

4. **Advantages of MLflow Integration**:

   - Standardized model serving interface
   - Access to model versions and metadata
   - Consistent experience across different model types
   - Easier deployment of complex models

Camera and Visualization
------------------------

### Starting the Camera

1. Click the "Classify from webcam stream" button on the UI to start the camera.
2. The button will toggle to "Stop streaming" when the camera is active.
3. The webcam feed will appear with pose detection overlay.

### Positioning for Best Results

1. Position yourself in the camera frame where your full body is visible.
2. Ensure adequate lighting for better pose detection accuracy.
3. Maintain a clear background if possible to improve detection.

### Viewing Classification Results

The interface will display classification output in real-time:

1. **Raw Classification Output** - JSON display of model predictions
2. **Status Indicators** - Visual feedback on classification status
3. **Frame Information** - Details about processed frames

### Visualization Options

The pose detection visualization includes:

1. **Keypoint Markers** - Points identifying detected body joints
2. **Skeleton Connections** - Lines connecting related joints
3. **Real-time Updates** - Continuous tracking of movement

Bluetooth Device Integration
--------------------------

### Connecting to Bluetooth Devices

StreamPoseML supports sending classification results to bluetooth devices:

1. Click the "Connect to Bluetooth" button in the UI.
2. Select your bluetooth device from the browser's device picker.
3. Authorize the connection in your browser.
4. The device status indicator will turn green when connected.

### Bluetooth Configuration

Expand the "Settings" section to configure bluetooth options:

1. **Service UUID** - Configure the bluetooth service UUID
2. **Characteristic UUID** - Configure the characteristic UUID for communication
3. **Response Strings** - Set message strings sent on positive/negative classifications

### Monitoring Bluetooth Communication

Expand the "Logs" section to view bluetooth communication:

1. **Connection Status** - Shows current connection status
2. **Sent/Received Data** - Displays data transmitted to and from the device
3. **Log History** - Complete history of bluetooth operations

*Note: Bluetooth integration currently only works in Chrome and Edge browsers.*

Performance and Optimization
---------------------------

### Real-time Performance Metrics

The web UI provides real-time performance information:

- **Frame Processing Rate** - Frames processed per second
- **Classification Latency** - Time between frame capture and classification
- **Model Prediction Time** - Time taken by the model to classify a frame

### Optimizing Performance

To improve real-time performance:

1. **Adjust Frame Window Size** - Decrease to reduce latency, increase for better accuracy
2. **Modify Prediction Frame Overlap** - Control how frequently the model makes predictions
3. **Use Simpler Models** - Less complex models typically run faster but may be less accurate

### Browser Recommendations

- Use Chrome or Edge for best performance and full feature support
- Ensure hardware acceleration is enabled in your browser
- Close unnecessary browser tabs and applications for smoother operation
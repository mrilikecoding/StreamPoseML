Usage
=====

Accessing the Web Application
----------------------------

After starting the application with ``make start``, the web interface will automatically open in your default browser. If it doesn't, you can access it at:

http://localhost:3000

The API is available at:

http://localhost:5001

Loading a Model
-------------

StreamPoseML supports two ways to load models: direct pickle files and MLflow-logged models.

### Option 1: Using Pickle Files

1. **Prepare your model**:
   
   Your trained model should be saved as a pickle file with a structure like:

   .. code-block:: python

      {
        "classifier": <your_trained_model implementing predict method>
      }

   The model should implement a ``predict`` method that takes an array of examples to classify.

2. **Place your model**:

   Copy your model file to the designated models directory. By default, this is:

   .. code-block:: none

      /path/to/StreamPoseML/models/

3. **Select the model in the UI**:

   - Open the web UI at http://localhost:3000
   - Go to Settings
   - Select your model from the dropdown
   - Click "Apply Changes"

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
   - Go to Settings > Models
   - Choose the "MLflow" model source
   - Select your model from the available MLflow models
   - Click "Apply Changes"

4. **Advantages of MLflow Integration**:

   - Standardized model serving interface
   - Access to model versions and metadata
   - Consistent experience across different model types
   - Easier deployment of complex models

Using the Webcam Interface
------------------------

1. **Start the webcam**:
   
   Click the "Start Camera" button on the UI.

2. **Position yourself**:
   
   Position yourself in the camera frame where your full body is visible.

3. **Observe classifications**:
   
   The interface will display:
   
   - Real-time pose detection overlay
   - Classification results
   - Confidence scores (if available)

4. **Control panel options**:
   
   - Toggle pose detection visualization
   - Adjust model parameters
   - View performance metrics

Bluetooth Device Integration
--------------------------

If your deployment includes bluetooth device integration:

1. Click "Connect Device" in the UI
2. Select your bluetooth device from the list
3. Authorize the connection in your browser
4. The device will now receive classification results

*Note: Bluetooth integration currently only works in Chrome and Edge browsers.*

Viewing Performance Metrics
-------------------------

The web UI provides real-time performance metrics:

- Frame processing rate
- Classification latency
- Model prediction time

These metrics can help you optimize your model and parameters for real-time performance.
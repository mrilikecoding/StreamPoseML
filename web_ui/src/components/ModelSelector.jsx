import React, { useState } from "react";

function ModelSelector({ setModel }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [frameWindowSize, setFrameWindowSize] = useState(30); // Initial value for frame window size
  const [frameWindowOverlap, setFrameWindowOverlap] = useState(5); // Initial value for frame window overlap

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleSubmit = async () => {
    const data = new FormData();
    data.append("file", selectedFile);
    data.append("frame_window_size", frameWindowSize);
    data.append("frame_window_overlap", frameWindowOverlap);

    try {
      const response = await fetch(
        import.meta.env.VITE_STREAM_POSE_ML_API_ENDPOINT + "/set_model",
        {
          body: data,
          method: "POST",
        }
      );
      
      if (!response.ok) {
        const errorData = await response.json();
        console.error("Error setting model:", errorData);
        setModel(`Error: ${errorData.result || response.statusText}`);
        return;
      }
      
      const responseData = await response.json();
      setModel(responseData.result);
    } catch (err) {
      console.error("Network error:", err);
      setModel(`Network error: ${err.message}`);
    }
  };

  return (
    <>
      <h2>Select Trained Model</h2>
      <h3>Frame Configuration</h3>
      <div className="text-xs">
        If you change these values, make sure to "set the model" again!
      </div>
      <div className="form-control mt-4">
        <label className="form-control w-full max-w-xs">
          <div className="label">
            <span className="label-text">
              <b>Frame Window Size</b>
            </span>
          </div>
          <input
            className="input input-bordered w-full max-w-xs"
            type="number"
            placeholder="Enter frame window size"
            value={frameWindowSize}
            onChange={(e) => setFrameWindowSize(parseInt(e.target.value) || 0)}
          />
          <div className="label">
            <span className="label-text-alt">
              This should match the model's expected number of input frames.
              This client will batch data from this number of frames as input
              for the model prediction as{" "}
              <code>{"joints.frame-{n}-{joint_name}.{axis}"}</code>
            </span>
          </div>
        </label>
      </div>
      <div className="form-control mt-4">
        <label className="form-control w-full max-w-xs">
          <div className="label">
            <span className="label-text">
              <b>Prediction Frame Overlap</b>
            </span>
          </div>
          <input
            className="input input-bordered w-full max-w-xs"
            type="number"
            placeholder="Enter frame window overlap"
            value={frameWindowOverlap}
            onChange={(e) =>
              setFrameWindowOverlap(parseInt(e.target.value) || 0)
            }
          />
          <div className="label">
            <span className="label-text-alt">
              How frequently to call the model. Positive values will result in
              overlapping frame window data based on the value. Negative values
              will result in a gap in frames between each prediction.
            </span>
          </div>
        </label>
      </div>
      <h3>Model File Selection</h3>
      <div className="bg-secondary collapse collapse-plus collapse w-full border border-base-300 bg-base-100 rounded-box">
        <input type="checkbox" className="peer" />
        <div className="collapse-title text-primary-content">
          <b>Model schema guidelines</b>
        </div>
        <div className="collapse-content text-primary-content">
          <p>
            This client assumes a model's expected columns are named a certain
            way, as this is how Mediapipe keypoint joints are serialized by
            StreamPoseML behind the scenes. For each frame <i>n</i> of an input
            data row, each column should be named according the following
            format:
          </p>
          <code>{"joints.frame-{n}-{joint_name}.{axis}"}</code>
          <p>
            Where <i>n</i> is the frame number and where each joint is named
            from the list:
          </p>
          <p>
            <code>
              nose, left_eye_inner, left_eye, left_eye_outer, right_eye_inner,
              right_eye, right_eye_outer, left_ear, right_ear, mouth_left,
              mouth_right,left_shoulder, right_shoulder, left_elbow,
              right_elbow, left_wrist, right_wrist, left_pinky, right_pinky,
              left_index, right_index, left_thumb, right_thumb, neck. left_hip,
              right_hip, mid_hip,left_knee, right_knee, left_ankle, right_ankle,
              left_heel, right_heel, left_foot_index, right_foot_index.
            </code>
          </p>
          <p>Each axis is one of:</p>
          <code>x, y, z</code>
        </div>
      </div>
      <h4 className="text-xs">
        This is either a pickle file generated via Stream Pose ML, or an MLFlow
        logged model (e.g. in gzip format with all needed assets an
        input_example file). Note, the model's expected input must conform to
        the this client's keypoint output format as described above.
      </h4>
      <div className="join">
        <input
          className="file-input file-input-bordered w-full max-w-xs join-item"
          type="file"
          onChange={handleFileChange}
        />
        <button
          className="btn btn-primary join-item"
          onClick={handleSubmit}
          disabled={!selectedFile}
        >
          Set Model
        </button>
      </div>
    </>
  );
}

export default ModelSelector;

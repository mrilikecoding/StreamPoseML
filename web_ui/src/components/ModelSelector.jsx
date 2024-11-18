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
    data.append("filename", selectedFile.name);
    data.append("file", selectedFile);
    data.append("frame_window_size", frameWindowSize); // Adding frame window size
    data.append("frame_window_overlap", frameWindowOverlap); // Adding frame window overlap

    fetch(import.meta.env.VITE_STREAM_POSE_ML_API_ENDPOINT + "/set_model", {
      body: data,
      method: "POST",
    })
      .then((res) => res.json())
      .then((data) => {
        setModel(data.result);
      })
      .catch((err) => console.error(err));
  };

  return (
    <>
      <h2>Select Trained Model</h2>
      <div className="collapse collapse-plus collapse w-full border border-base-300 bg-base-100 rounded-box">
        <input type="checkbox" className="peer" />
        <div className="collapse-title text-primary-content">
          Model schema guidelines...
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
      <div className="form-control mt-4">
        <label className="form-control w-full max-w-xs">
          <div className="label">
            <span className="label-text">Frame Window Size</span>
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
              This should match the model's expected number of input frames
            </span>
          </div>
        </label>
      </div>
      <div className="form-control mt-4">
        <label className="form-control w-full max-w-xs">
          <div className="label">
            <span className="label-text">Prediction Frame Overlap</span>
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
              This value determines how frequently to call the model. Positive
              values will result in overlapping frame window data based on the
              value above. Negative values will result in a gap in frames
              between subsquent predictions. For example, -50 will result in the
              model getting invoked every frame_window - -50 frames. So if there
              is a frame window of 30, 30 - -50 will result in a 20 frame gap
              between each prediction. An overlap of 5 and a frame window of 30
              will result in predictions every 25 frames.
            </span>
          </div>
        </label>
      </div>
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

Segmentation Strategies Explained
=============================

Understanding Segmentation in StreamPoseML
----------------------------------------

Segmentation strategies are a crucial part of preparing pose data for machine learning. In StreamPoseML, segmentation refers to how individual video frames are grouped together to create meaningful training examples. The right segmentation strategy can dramatically improve your model's ability to recognize movements accurately.

This guide provides a detailed explanation of each segmentation strategy available in StreamPoseML and when to use them.

Why Segmentation Matters
----------------------

Movement happens over time, not in a single frame. When analyzing human motion, you'll typically want to consider sequences of frames rather than individual snapshots. Segmentation strategies help you:

1. **Define meaningful movement units** - Group frames that represent a complete movement or pose
2. **Structure temporal data** - Organize time-series data for effective machine learning
3. **Balance detail and abstraction** - Control the granularity of movement representation
4. **Manage dataset size** - Create appropriate training examples without overwhelming your model

Available Segmentation Strategies
------------------------------

StreamPoseML offers five distinct segmentation strategies, each suitable for different use cases:

1. **None** - Each frame is its own data point
2. **Split on Label** - Frames are grouped based on shared labels
3. **Window** - Fixed-size windows of consecutive frames
4. **Flatten into Columns** - Windows of frames flattened into single-row features
5. **Flatten on Example** - Combination of splitting on labels and flattening

Let's explore each strategy in detail:

Strategy 1: None
--------------

**What it does:**
Each individual frame is treated as a separate training example, with no temporal context.

**In the code:**
```python
def segment_all_frames(self, dataset: "Dataset") -> list[LabeledClip]:
    """This method creates a list where every frame is its own LabeledClip."""
    segmented_data = []
    if self.include_unlabeled_data:
        for frame in sum(dataset.all_frames, []):
            segmented_data.append(LabeledClip(frames=[frame]))
    else:
        for frame in sum(dataset.labeled_frames, []):
            segmented_data.append(LabeledClip(frames=[frame]))
    return segmented_data
```

**Best for:**
- Static pose classification (like yoga poses)
- When temporal relationships don't matter
- Simple classification tasks
- Very fine-grained analysis

**Example usage:**
```python
formatted_dataset = db.format_dataset(
    dataset=dataset,
    include_angles=True,
    include_distances=True,
    segmentation_strategy="none"  # Each frame becomes a separate training example
)
```

**Visualization:**
Original video: [f1, f2, f3, f4, f5]
Segmentation result: [[f1], [f2], [f3], [f4], [f5]]

Strategy 2: Split on Label
-----------------------

**What it does:**
Groups consecutive frames that share the same label value into a single segment. This creates sequences that represent complete labeled movements.

**In the code:**
```python
def split_on_label(self, dataset: "Dataset", flatten_into_columns: bool = False) -> list[LabeledClip]:
    """Split a Dataset's labeled_frame list into segments sharing the same label."""
    # For each video's frames
    for video in labeled_frame_videos:
        # Group consecutive frames with the same label value
        for i, frame in enumerate(video):
            if (i + 1) == len(video):
                # Handle last frame
                segmented_frames[segment_counter].append(frame)
            elif (video[i + 1][segment_splitter_label] == frame[segment_splitter_label]):
                # Same label as next frame - add to current segment
                if segment_counter in segmented_frames:
                    segmented_frames[segment_counter].append(frame)
                else:
                    segmented_frames[segment_counter] = [frame]
            else:
                # Label changes after this frame - start a new segment
                segment_counter += 1
```

**Best for:**
- Movements with clear start/end points
- Analyzing complete movement sequences
- When labels naturally segment the video (e.g., "left step", "right step")
- Capturing variable-length movements

**Example usage:**
```python
formatted_dataset = db.format_dataset(
    dataset=dataset,
    include_angles=True,
    include_distances=True,
    segmentation_strategy="split_on_label",  # Group frames with same label
    segmentation_splitter_label="step_type"  # Label that defines segments
)
```

**Visualization:**
Original video with labels: [(f1,"L"), (f2,"L"), (f3,"R"), (f4,"R"), (f5,"L")]
Segmentation result: [[f1,f2], [f3,f4], [f5]]

Strategy 3: Window
---------------

**What it does:**
Creates fixed-size windows of consecutive frames where the last frame has a specified label. This is ideal for capturing movements with consistent duration.

**In the code:**
```python
def split_on_window(self, dataset: "Dataset") -> list[LabeledClip]:
    """Segment video frame data based on a fixed window size."""
    # For each video
    for video in all_frame_videos:
        # For each possible window position
        for i, frame in enumerate(video):
            # Skip until we have enough frames for a complete window
            if i < segment_window_size:
                continue
            # Check if the last frame has the required label
            elif (i % segment_window_size == 0 and video[i][segment_window_label] is not None):
                # Create the window
                frame_segment = []
                for j in range(1 + i - segment_window_size, i + 1):
                    frame_segment.append(video[j])
                segmented_frames[segment_counter] = frame_segment
                segment_counter += 1
```

**Best for:**
- Fixed-duration movements
- Regular sampling of video for consistent input sizes
- When you need exactly N frames per example
- Real-time applications with fixed processing windows

**Example usage:**
```python
formatted_dataset = db.format_dataset(
    dataset=dataset,
    include_angles=True,
    include_distances=True,
    segmentation_strategy="window",           # Use fixed-size windows
    segmentation_window=10,                   # Each window contains 10 frames
    segmentation_window_label="movement_type" # Label to check on last frame
)
```

**Visualization:**
Original video: [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]
With window size 5: [[f1,f2,f3,f4,f5], [f6,f7,f8,f9,f10]]

Strategy 4: Flatten into Columns
----------------------------

**What it does:**
Similar to window strategy, but transforms the window's frames into a single row with separate columns for each frame's features. This converts temporal data into a wide, single-row representation.

**In the code:**
```python
def flatten_into_columns(self, dataset: "Dataset") -> list[LabeledClip]:
    """Segment video data based on a window and flatten into frame columns."""
    # Similar to window strategy for selecting frames
    # Then for each window:
    flattened = self.flatten_segment_into_row(frame_segment=frame_segment)
    segmented_frames[segment_counter] = [flattened]
```

**The flattening process:**
```python
def flatten_segment_into_row(frame_segment: list):
    """Flatten a list of frames into a single row object."""
    # Copy metadata from last frame
    flattened = {key: value for key, value in frame_segment[-1].items() 
                if (isinstance(value, str) or value is None)}
    flattened["data"] = {}
    
    # For each frame in the segment
    for i, frame in enumerate(frame_segment):
        frame_data = frame["data"]
        # For each data type (angles, distances, etc.)
        for key, value in frame_data.items():
            if key not in flattened["data"]:
                flattened["data"][key] = {}
            # Create frame-specific column names
            if isinstance(value, dict):
                for k, v in value.items():
                    flattened["data"][key][f"frame-{i+1}-{k}"] = v
            else:
                flattened["data"][key] = value
```

**Best for:**
- Creating fixed-width feature vectors for traditional ML models
- When you need to preserve temporal information but use non-sequential models
- Feature engineering with explicit frame-by-frame values
- Combining with feature selection to identify key frames/positions

**Example usage:**
```python
formatted_dataset = db.format_dataset(
    dataset=dataset,
    include_angles=True,
    include_distances=True,
    segmentation_strategy="flatten_into_columns",  # Create wide feature rows
    segmentation_window=5,                         # Use 5 frame windows
    segmentation_window_label="weight_transfer_type" # Label for the window
)
```

**Visualization:**
Original data:
```
frame1: {angles: {elbow: 90}, distances: {hand_to_hip: 45}}
frame2: {angles: {elbow: 100}, distances: {hand_to_hip: 40}}
```

Flattened result:
```
{angles: {frame-1-elbow: 90, frame-2-elbow: 100}, 
 distances: {frame-1-hand_to_hip: 45, frame-2-hand_to_hip: 40}}
```

Strategy 5: Flatten on Example
--------------------------

**What it does:**
A combination of "split on label" and "flatten into columns". It first groups frames by their shared label, then flattens each group into a single row with frame-specific columns. This gives you natural movement segments with a fixed-width representation.

**In the code:**
```python
def flatten_on_example(self, dataset: "Dataset") -> list[LabeledClip]:
    """Segment based on label and flatten data into frame columns."""
    # This calls split_on_label with flatten_into_columns=True
    return self.split_on_label(dataset=dataset, flatten_into_columns=True)
```

**Best for:**
- Complete movement cycles with variable length
- Preserving movement boundaries while creating fixed-width features
- Combining semantically meaningful segments with traditional ML models
- Feature engineering on natural movement units

**Example usage:**
```python
formatted_dataset = db.format_dataset(
    dataset=dataset,
    include_angles=True,
    include_distances=True,
    segmentation_strategy="flatten_on_example",      # Split by label and flatten
    segmentation_splitter_label="step_type",         # Label defining segments
    segmentation_window=10,                          # Use last 10 frames if segment larger
    segmentation_window_label="weight_transfer_type" # Additional label info
)
```

**Visualization:**
Original video with labels: [(f1,"L"), (f2,"L"), (f3,"R"), (f4,"R")]
Segmented: [[f1,f2], [f3,f4]]
Flattened: [
  {angles: {frame-1-elbow: 90, frame-2-elbow: 95}, distances: {...}},
  {angles: {frame-1-elbow: 100, frame-2-elbow: 105}, distances: {...}}
]

Choosing the Right Segmentation Strategy
-------------------------------------

Each segmentation strategy offers different trade-offs. Here are some guidelines for choosing:

| Strategy | When to Use | Considerations |
|----------|-------------|----------------|
| **None** | Static poses, simple classification | No temporal information |
| **Split on Label** | Complete movement cycles, semantic boundaries | Variable length segments |
| **Window** | Fixed duration movements, consistent inputs | May cut across movement boundaries |
| **Flatten into Columns** | Traditional ML models needing fixed-width inputs | Creates high-dimensional data |
| **Flatten on Example** | Balance between semantic meaning and fixed-width | Most complex, but often most effective |

Practical Example
---------------

Here's a practical example showing how to use different segmentation strategies for a dance step classification task:

```python
# 1. Frame-by-frame analysis (no segmentation)
formatted_dataset_1 = db.format_dataset(
    dataset=dataset,
    include_angles=True,
    include_distances=True,
    segmentation_strategy="none"
)

# 2. Complete dance steps (split on label)
formatted_dataset_2 = db.format_dataset(
    dataset=dataset,
    include_angles=True,
    include_distances=True,
    segmentation_strategy="split_on_label",
    segmentation_splitter_label="step_type"  # "left_step" or "right_step"
)

# 3. Fixed 30-frame windows (about 1 second of video)
formatted_dataset_3 = db.format_dataset(
    dataset=dataset,
    include_angles=True,
    include_distances=True,
    segmentation_strategy="window",
    segmentation_window=30,
    segmentation_window_label="weight_transfer_type"
)

# 4. Dance steps flattened into feature columns
formatted_dataset_4 = db.format_dataset(
    dataset=dataset,
    include_angles=True,
    include_distances=True,
    segmentation_strategy="flatten_on_example",
    segmentation_splitter_label="step_type",
    segmentation_window=10,  # Use last 10 frames if step longer
    segmentation_window_label="weight_transfer_type"
)
```

Advanced Usage
-----------

For more advanced use cases, you can combine segmentation strategies with other dataset formatting options:

```python
# Advanced example combining multiple options
formatted_dataset = db.format_dataset(
    dataset=dataset,
    # Feature selection
    include_angles=True,          # Include joint angles
    include_distances=True,       # Include distances between joints
    include_normalized=True,      # Include normalized coordinates
    include_joints=False,         # Exclude raw joint positions
    include_z_axis=False,         # Exclude z-axis data
    
    # Data processing
    pool_frame_data_by_clip=False, # Don't aggregate features across frames
    decimal_precision=4,          # Round values to 4 decimal places
    include_unlabeled_data=False, # Exclude unlabeled frames
    
    # Segmentation strategy
    segmentation_strategy="flatten_on_example",
    segmentation_splitter_label="step_type",
    segmentation_window=10,
    segmentation_window_label="weight_transfer_type"
)
```

Conclusion
--------

Choosing the right segmentation strategy is essential for effective movement classification. By understanding the available strategies in StreamPoseML, you can create more meaningful training examples that better capture the temporal nature of human movement.

Experiment with different strategies to see which works best for your specific use case. Often, the right segmentation strategy can improve model performance more than complex model architectures or hyperparameter tuning.
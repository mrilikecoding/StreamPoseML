OpenPose and MediaPipe Integration
==============================

Understanding Pose Format Compatibility
------------------------------------

StreamPoseML primarily uses MediaPipe's BlazePose for pose detection, but also provides compatibility with the OpenPose format, which is widely used in research and existing datasets. This document explains how StreamPoseML handles the conversion between these two pose estimation systems.

Why Pose Format Compatibility Matters
----------------------------------

Researchers and developers often have existing datasets or models based on specific pose estimation formats. By providing compatibility between MediaPipe's BlazePose and OpenPose, StreamPoseML enables:

1. **Migration of existing projects** - Use StreamPoseML with data originally collected using OpenPose
2. **Comparison with prior research** - Compare your results with studies that used OpenPose
3. **Broader ecosystem compatibility** - Integrate with tools and libraries designed for either format

How the Conversion Works
---------------------

StreamPoseML provides the ``OpenPoseMediapipeTransformer`` class that handles the translation between the two formats. The transformer primarily works by:

1. **Creating intermediate joints** - Some joints in OpenPose don't directly exist in BlazePose and vice versa
2. **Mapping equivalent keypoints** - Identifying which keypoints correspond between the two systems
3. **Computing derived vectors** - Creating vectors that represent relationships between keypoints
4. **Calculating equivalent measurements** - Ensuring angles and distances are comparable

The Transformation Process
-----------------------

When you use OpenPose compatibility features, here's what happens behind the scenes:

1. BlazePose detects keypoints from the video/image
2. The ``OpenPoseMediapipeTransformer`` adds derived joints like "neck" and "mid_hip" by averaging other keypoints
3. The transformer creates vectors between joints that match OpenPose's Body-25 model
4. These vectors are used to calculate angles and distances that match OpenPose's measurements

Key Differences Between BlazePose and OpenPose
-------------------------------------------

Here are the main differences between the two pose estimation systems:

| Aspect | BlazePose (MediaPipe) | OpenPose |
|--------|------------------------|----------|
| **Number of keypoints** | 33 full-body landmarks | Various models (15, 18, 25 points) |
| **Key features** | More detailed hand tracking | More established in research |
| **Reference model** | Full-body landmark model | Body-25 model (in StreamPoseML) |
| **Coordinate system** | Normalized to image dimensions | Pixel coordinates |

Joint Mapping
-----------

Here's how some key joints map between the two systems:

| OpenPose (Body-25) | BlazePose (MediaPipe) | Notes |
|--------------------|------------------------|-------|
| Nose | NOSE | Direct mapping |
| Neck | Not present directly | Created by averaging shoulders |
| Right Shoulder | RIGHT_SHOULDER | Direct mapping |
| Right Elbow | RIGHT_ELBOW | Direct mapping |
| Right Wrist | RIGHT_WRIST | Direct mapping |
| Left Shoulder | LEFT_SHOULDER | Direct mapping |
| Left Elbow | LEFT_ELBOW | Direct mapping |
| Left Wrist | LEFT_WRIST | Direct mapping |
| Mid Hip | Not present directly | Created by averaging hips |
| Right Hip | RIGHT_HIP | Direct mapping |
| Right Knee | RIGHT_KNEE | Direct mapping |
| Right Ankle | RIGHT_ANKLE | Direct mapping |
| Left Hip | LEFT_HIP | Direct mapping |
| Left Knee | LEFT_KNEE | Direct mapping |
| Left Ankle | LEFT_ANKLE | Direct mapping |
| Right Big Toe | RIGHT_FOOT_INDEX | Approximate mapping |
| Right Small Toe | Not present | No good equivalent |
| Right Heel | RIGHT_HEEL | Direct mapping |
| Left Big Toe | LEFT_FOOT_INDEX | Approximate mapping |
| Left Small Toe | Not present | No good equivalent |
| Left Heel | LEFT_HEEL | Direct mapping |

How to Use OpenPose Compatibility
------------------------------

The OpenPose compatibility feature is automatically applied when you use certain features in StreamPoseML. Here's how you can take advantage of it:

### 1. Creating OpenPose-Compatible Vectors and Joints

The ``create_openpose_joints_and_vectors`` method is called internally when needed:

```python
from stream_pose_ml.blaze_pose.openpose_mediapipe_transformer import OpenPoseMediapipeTransformer

# This typically happens automatically in the BlazePoseFrame class
transformer = OpenPoseMediapipeTransformer()
transformer.create_openpose_joints_and_vectors(frame)
```

### 2. Using OpenPose-Style Distance Measurements

You can access the OpenPose distance definitions:

```python
from stream_pose_ml.blaze_pose.openpose_mediapipe_transformer import OpenPoseMediapipeTransformer

# Get the mapping of OpenPose distance definitions to StreamPoseML vectors
distance_map = OpenPoseMediapipeTransformer.open_pose_distance_definition_map()

# Example of what you'd get:
# {'nose_to_plumb_line': ('nose', 'plumb_line'),
#  'neck_to_plumb_line': ('neck', 'plumb_line'),
#  ...}
```

### 3. Using OpenPose-Style Angle Measurements

Similarly for angles:

```python
from stream_pose_ml.blaze_pose.openpose_mediapipe_transformer import OpenPoseMediapipeTransformer

# Get the mapping of OpenPose angle definitions to StreamPoseML vectors
angle_map = OpenPoseMediapipeTransformer.open_pose_angle_definition_map()
```

Important Implementation Details
-----------------------------

When examining the code, it's clear that:

1. **StreamPoseML converts BlazePose to OpenPose format**, not the other way around
2. The conversion creates **derived joints** not directly present in BlazePose (like "neck")
3. The conversion focuses on the **Body-25 model** from OpenPose
4. Special attention is paid to the **plumb line** concept (vertical reference line)
5. Some OpenPose keypoints (like small toes) have **no direct equivalent** in BlazePose

The Plumb Line Concept
-------------------

One important concept in the conversion is the "plumb line" - a vertical reference line used for many angle measurements:

- In OpenPose's Body-25 model, this is represented by the vector between joints 25 and 26 (neck and mid-hip)
- In StreamPoseML's BlazePose integration, this is created by calculating the vector between:
  - The neck point (average of left and right shoulders)
  - The mid-hip point (average of left and right hips)

This plumb line serves as a reference for many angle and distance measurements in both systems.

Example Code
----------

Here's a comprehensive example that shows how to use OpenPose-compatible measurements:

```python
from stream_pose_ml.blaze_pose.blaze_pose_frame import BlazePoseFrame
from stream_pose_ml.blaze_pose.mediapipe_client import MediaPipeClient
import cv2

# Initialize the MediaPipe client
mpc = MediaPipeClient()

# Process an image and get keypoints
image = cv2.imread('pose_image.jpg')
keypoint_results = mpc.process_image(image)

# Create a BlazePoseFrame from the keypoints
# This automatically creates OpenPose-compatible joints and vectors
frame = BlazePoseFrame(
    keypoint_results=keypoint_results,
    include_geometry=True  # This triggers the OpenPose conversion
)

# Now you can access OpenPose-style measurements
# For example, get the angle between the torso and the vertical plumb line
torso_verticality = frame.angles.get("neck_mid_hip_to_plumb_line")
print(f"Torso angle from vertical: {torso_verticality} degrees")

# Or calculate how far the right hand is from the plumb line
right_hand_offset = frame.distances.get("right_wrist_to_plumb_line")
print(f"Right hand distance from center line: {right_hand_offset} pixels")
```

Limitations
---------

There are a few limitations to be aware of when using the OpenPose compatibility features:

1. **Not all keypoints map perfectly** - Some OpenPose keypoints have no direct equivalent in BlazePose
2. **Some calculations are approximations** - The two systems use different detection methods
3. **No backwards conversion** - StreamPoseML doesn't convert from OpenPose to BlazePose format
4. **Limited to Body-25** - Support focuses on OpenPose's Body-25 model, not other variants

Conclusion
--------

The OpenPose compatibility in StreamPoseML provides a valuable bridge between different pose estimation ecosystems. By understanding how this conversion works, you can:

- Leverage existing datasets and research
- Compare results across different pose estimation systems
- Apply StreamPoseML to a wider range of use cases

This flexibility makes StreamPoseML more versatile and useful in both research and practical applications.
from stream_pose_ml.learning.labeled_clip import LabeledClip
from stream_pose_ml.serializers.labeled_frame_serializer import LabeledFrameSerializer
from stream_pose_ml.learning import temporal_feature_pooling as tfp


class LabeledClipSerializer:
    include_joints: bool
    include_angles: bool
    include_distances: bool
    include_normalized_points: bool
    include_non_normalized_points: bool
    include_z_axis: bool
    decimal_precision: int
    angle_whitelist: list
    joint_whitelist: list
    distance_whitelist: list

    def __init__(
        self,
        include_joints: bool = False,
        include_angles: bool = True,
        include_distances: bool = True,
        include_non_normalized_points: bool = True,
        include_normalized: bool = True,
        include_z_axis: bool = False,
        # TODO angles / distances / joints to include (NOT IMPLEMENTED)
        angle_whitelist: list = [],
        distance_whitelist: list = [],
        joint_whitelist: list = [],
        # pooling options if pooling the temporal data
        pool_avg: bool = True,
        pool_std: bool = True,
        pool_max: bool = True,
    ):
        self.include_joints = include_joints
        self.include_angles = include_angles
        self.include_distances = include_distances
        self.include_non_normalized_points = include_non_normalized_points
        self.include_normalized = include_normalized
        self.include_z_axis = include_z_axis
        self.angle_whitelist = angle_whitelist
        self.distance_whitelist = distance_whitelist
        self.joint_whitelist = joint_whitelist
        self.pool_avg = pool_avg
        self.pool_std = pool_std
        self.pool_max = pool_max

    def serialize(
        self, labeled_clip: LabeledClip, pool_rows: bool = True
    ) -> dict | list[dict]:
        """This method serializes a LabeledClip object

        Args:
            labeled_clip: LabeledClip
                a clip of labeled data - note only the last frame in the clip needs the label really
                this is because sometimes we'll want to include unlabeled frames as part of the entire
                set of frames, for example when doing temporal pooling over a frame window where the last
                frame has the label for the preceding x frames in the window. Other times we may want
                variable length clips. This method doesn't really care.
        Returns:
            pooled_clip_data: dict
                a dictionary of various pooled temporal features if self.pool_frame_data is True
            OR
            frame_data_list: list[dict]
                a list of serialized frame data for every frame in the clip
        """
        clip_data = {"frame_length": len(labeled_clip.frames)}
        frame_rows = []

        frame_serializer = LabeledFrameSerializer(
            include_angles=self.include_angles,
            include_distances=self.include_distances,
            include_joints=self.include_joints,
            include_normalized=self.include_normalized,
            include_z_axis=self.include_z_axis,
        )
        for frame in labeled_clip.frames:
            frame_rows.append(frame_serializer.serialize(frame=frame))

        if pool_rows:
            # If pooling data, clips should be labeled according to their last frame
            # Clips should always have a label on the last frame
            # TODO remove hard coding here
            meta_keys = ["video_id", "weight_transfer_type", "step_type"]
            for key in meta_keys:
                clip_data[key] = frame_rows[-1][key]
            if self.include_angles:
                angles = [frame["angles"] for frame in frame_rows]
                if self.pool_avg:
                    clip_data["angles_avg"] = tfp.compute_average_value(angles)
                if self.pool_max:
                    clip_data["angles_max"] = tfp.compute_max(angles)
                if self.pool_std:
                    clip_data["angles_std"] = tfp.compute_standard_deviation(angles)
            if self.include_distances:
                distances = [frame["distances"] for frame in frame_rows]
                if self.pool_avg:
                    clip_data["distances_avg"] = tfp.compute_average_value(distances)
                if self.pool_max:
                    clip_data["distances_max"] = tfp.compute_max(distances)
                if self.pool_std:
                    clip_data["distances_std"] = tfp.compute_standard_deviation(
                        distances
                    )
            return clip_data
        else:
            return frame_rows

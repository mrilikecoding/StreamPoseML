from pose_parser.learning.labeled_clip import LabeledClip
from pose_parser.serializers.labeled_frame_serializer import LabeledFrameSerializer
from pose_parser.learning import temporal_feature_pooling as tfp


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

    def __init__(self):
        self.include_joints = False
        self.include_angles = True
        self.include_distances = False
        self.include_non_normalized_points = True
        self.include_normalized_points = False
        self.include_z_axis = False
        # TODO angles / distances / joints to include (NOT IMPLEMENTED)
        self.angle_whitelist = []
        self.distance_whitelist = []
        self.joint_whitelist = []
        # pooling options if pooling the temporal data
        self.pool_avg = True
        self.pool_std = True
        self.pool_max = True
        # report the std of the frame values
        self.include_std = True

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
            include_angles=self.include_angles
        )
        for frame in labeled_clip.frames:
            frame_rows.append(frame_serializer.serialize(frame=frame))
        if pool_rows:
            # if self.include_distances:
            # distances = self.serialize_distances(frame["data"]["distances"])
            # joints = self.serialize_joints(frame["data"]["joints"])
            if self.include_angles:
                angles = [self.serialize_angles(frame['data']['angles']) for frame in frame_rows]
                if self.pool_avg:
                    clip_data["angles_avg"] = tfp.compute_average_value(angles)
                if self.pool_max:
                    clip_data["angles_max"] = tfp.compute_max(angles)
                if self.pool_std:
                    clip_data["angles_std"] = tfp.compute_std(angles)
            return clip_data
        else:
            return frame_rows

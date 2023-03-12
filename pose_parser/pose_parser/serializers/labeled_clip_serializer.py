from pose_parser.learning.labeled_clip import LabeledClip


class LabeledClipSerializer:
    def __init__(self):
        self.include_joints = False
        self.include_angle = True
        self.include_distances = False
        self.include_non_normalized_points = True
        self.include_normalized_points = False
        self.include_z_axis = False
        self.decimal_precision = 4
        self.angle_whitelist = []
        self.distance_whitelist = []
        self.joint_whitelist = []
        # collapse values from all frames into their average value
        self.average_all_frame_values = True
        # report the std of the frame values
        self.include_std = True

    def serialize(self, labeled_clip: LabeledClip):
        # TODO here, write all the columns we want to inclide in the finale dataset for a clip
        # this should include temporal statistics computed across all frames
        data = {
            "step_type": labeled_clip.frames[-1]["step_type"],
            "weight_transfer_type": labeled_clip.frames[-1]["weight_transfer_type"],
            "frame_length": len(labeled_clip.frames),
        }

        angles = []
        for frame in labeled_clip.frames:
            angles.append(self.serialize_angles(frame["data"]["angles"]))
            # distances = self.serialize_distances(frame["data"]["distances"])
            # joints = self.serialize_joints(frame["data"]["joints"])

        data["angles_avg"] = self.compute_average_value(
            angles, decimal_precision=self.decimal_precision
        )
        data["angles_std"] = self.compute_standard_deviation(
            angles, decimal_precision=self.decimal_precision
        )

        return data

    @staticmethod
    def compute_standard_deviation(
        dict_list: list[dict], decimal_precision: int
    ) -> dict:
        std_dict = {}
        for key in dict_list[0].keys():
            mean = sum(d[key] for d in dict_list) / len(dict_list)
            std_dict[key] = round(
                sum((d[key] - mean) ** 2 for d in dict_list) / len(dict_list),
                decimal_precision,
            )

        return std_dict

    @staticmethod
    def compute_average_value(dict_list: list[dict], decimal_precision: int) -> dict:
        mean_dict = {}
        for key in dict_list[0].keys():
            mean_dict[key] = round(
                sum(d[key] for d in dict_list) / len(dict_list), decimal_precision
            )
        return mean_dict

    @staticmethod
    def serialize_angles(angles):
        angle_dictionary = {}
        for angle, data in angles.items():
            angle_dictionary[f"{angle}_angle_2d_degrees"] = data["angle_2d_degrees"]
        return angle_dictionary

    @staticmethod
    def serialize_distance(distances):
        return {}

    @staticmethod
    def serialize_joints(joints):
        return {}

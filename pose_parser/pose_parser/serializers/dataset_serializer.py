from pose_parser.serializers.labeled_clip_serializer import LabeledClipSerializer


class LabeledFrameSerializer:
    def __init__(self, decimal_precision: int = 4, include_angles: bool = True, include_distances: bool = False, include_joints: bool = False) -> None:
        self.decimal_precision = 4
        self.include_angles = include_angles
        self.include_distances = include_distances
        self.include_joints = include_joints

    def serialize(self, frame: dict):
        """
        TODO - refactor this to be use in the LabeledClip serializer
        TODO - add options for including angles etc... returning some specific hard coded stuff for now
        """
        angles = self.serialize_angles(frame["data"]["angles"], self.decimal_precision)
        row = {
            "video_id": frame["video_id"],
            "step_frame_id": 0
            if (frame["weight_transfer_type"] and frame["step_type"])
            else "NULL",
            "frame_number": frame["data"]["frame_number"],
            "weight_transfer_type": frame["weight_transfer_type"]
            if frame["weight_transfer_type"]
            else "NULL",
            "step_type": frame["step_type"] if frame["step_type"] else "NULL",
            "angles": angles,
        }

        return row

    @staticmethod
    def serialize_angles(angles, decimal_precision):
        angle_dictionary = {}
        for angle, data in angles.items():
            angle_dictionary[f"{angle}_angle_2d_degrees"] = round(
                data["angle_2d_degrees"], decimal_precision
            )
        return angle_dictionary


class DatasetSerializer:
    def __init__(self, combine_rows: bool = True):
        self.combine_rows = combine_rows

    def serialize(self, dataset: list):
        """
        Args:
            dataset: list[LabeledClips]

        Returns:
            clips: list[dict]
                A list of serialized labeled clips
        """
        rows = []
        frame_serializer = LabeledFrameSerializer()
        for clip in dataset:
            if self.combine_rows:
                rows.append(LabeledClipSerializer().serialize(clip))
            else:
                for frame in clip.frames:
                    # add one to counter since frames are 1 indexed
                    rows.append(frame_serializer.serialize(frame))

        sorted_rows = sorted(rows, key=lambda r: (r["video_id"], r["frame_number"]))
        clip_frame_counter = 1
        for i, row in enumerate(sorted_rows):
            if i + 1 == len(sorted_rows):
                break
            if row["step_type"] is "NULL" and row["weight_transfer_type"] is "NULL":
                row["step_frame_id"] = "NULL"
            else:
                row["step_frame_id"] = clip_frame_counter
            if sorted_rows[i + 1]["step_type"] != row["step_type"]:
                clip_frame_counter = 0
            elif sorted_rows[i + 1]["video_id"] != row["video_id"]:
                clip_frame_counter = 0
            clip_frame_counter += 1

        return sorted_rows

from pose_parser.serializers.labeled_clip_serializer import LabeledClipSerializer


class DatasetSerializer:
    """This class serializes a timeseries dataset by either outputting rows for each frame (for PCA perhaps) or combines clips (with temmporal pooling)
    An input dataset should be a list of lists of frame data dictionaries that correspond to some label or labels.
    Each frame_data list can be of different length or correspond to a window of temporal data where the last frame's label represents the label of the window
    """

    pool_rows: bool

    def __init__(
        self,
        pool_rows: bool = True,
    ):
        """Initialize the serializer with options

        Args:
            pool_rows: bool
                whether to use temporal pooling to aggregate temporal features across all frames for each clip
        Returns:
            dataset: list[dict]
                a list of data dictionaries either corresponding to a pooled clip data or individual frames
        """

        self.pool_rows = pool_rows

    def serialize(self, dataset: list) -> list[dict]:
        """
        Args:
            dataset: list[LabeledClips]

        Returns:
            clips: list[dict]
                A list of serialized labeled clips
        """
        rows = []
        clip_serializer = LabeledClipSerializer()
        for clip in dataset:
            if self.pool_rows:
                # append pooled frame data to rows
                rows.append(
                    clip_serializer.serialize(labeled_clip=clip, pool_rows=True)
                )
            else:
                # append list of frame data to rows
                clip_data = clip_serializer.serialize(
                    labeled_clip=clip, pool_rows=False
                )
                rows = rows + clip_data

        if self.pool_rows:
            return rows
        # if not pooling data, it's helpful to sort by video_id and frame number
        # then determine the step number relative to the labeled sequence within the video
        else:
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
        e

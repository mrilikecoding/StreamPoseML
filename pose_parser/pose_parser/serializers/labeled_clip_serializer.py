from pose_parser.learning.labeled_clip import LabeledClip


class LabeledClipSerializer:
    @staticmethod
    def serialize(labeled_clip: LabeledClip, opts={}):
        # here, write all the columns we want to inclide in the finale dataset for a clip
        # this should include temporal statistics computed across all frames
        data = {
            "step_type": labeled_clip.frames[-1]["step_type"],
            "weight_transfer_type": labeled_clip.frames[-1]["weight_transfer_type"],
            "frame_length": len(labeled_clip.frames),
        }

        include_angles = opts["include_angles"]
        include_distances = opts["include_distances"]
        include_normalized = opts["include_normalized"]
        include_z = opts["include_z"]

        return data

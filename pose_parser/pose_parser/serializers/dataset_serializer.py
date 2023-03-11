from pose_parser.serializers.labeled_clip_serializer import LabeledClipSerializer


class DatasetSerializer:
    @staticmethod
    def serialize(dataset: list):
        """
        Args:
            dataset: list[LabeledClips]
        """
        clips = []
        for clip in dataset:
            clips.append(LabeledClipSerializer().serialize(clip))

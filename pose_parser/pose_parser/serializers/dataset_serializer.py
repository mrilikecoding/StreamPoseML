from pose_parser.serializers.labeled_clip_serializer import LabeledClipSerializer


class DatasetSerializer:
    @staticmethod
    def serialize(dataset: list):
        """
        Args:
            dataset: list[LabeledClips]
        
        Returns:
            clips: list[dict]
                A list of serialized labeled clips
        """
        clips = []
        for clip in dataset:
            clips.append(LabeledClipSerializer().serialize(clip))
        return clips

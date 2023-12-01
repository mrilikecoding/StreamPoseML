from stream_pose_ml.blaze_pose.blaze_pose_sequence import BlazePoseSequence
from stream_pose_ml.serializers.blaze_pose_frame_serializer import (
    BlazePoseFrameSerializer,
)


class BlazePoseSequenceSerializer:
    @staticmethod
    def serialize(
        blaze_pose_sequence: BlazePoseSequence, key_off_frame_number: bool = False
    ) -> dict:
        """This method is responsible for returning a formatted data object
        for the passed blaze pose sequence.

        Args:
            blaze_pose_sequence: BlazePoseSequence
                a BlazePoseSequence object
            key_off_frame_number: bool
                if True, the frame data will a dictionary keyed off frame number

        Returns:
            data: dict |
                A dict containing the sequence name and its serialized frame data
        """
        if key_off_frame_number:
            frame_data = {
                frame.frame_number: BlazePoseFrameSerializer.serialize(frame)
                for frame in blaze_pose_sequence.frames
            }
        else:
            frame_data = [
                BlazePoseFrameSerializer.serialize(frame)
                for frame in blaze_pose_sequence.frames
            ]

        return {
            "name": blaze_pose_sequence.name,
            "type": "BlazePoseSequence",
            "frames": frame_data,
        }

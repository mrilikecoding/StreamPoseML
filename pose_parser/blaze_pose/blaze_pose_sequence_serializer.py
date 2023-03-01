from pose_parser.blaze_pose.blaze_pose_sequence import BlazePoseSequence
from pose_parser.blaze_pose.blaze_pose_frame_serializer import BlazePoseFrameSerializer


class BlazePoseSequenceSerializer:
    @staticmethod
    def serialize(blaze_pose_sequence: BlazePoseSequence):
        """
        This method is responsible for returning a formatted data object
        for the passed blaze pose sequence

        Parameters
        ---------
            blaze_pose_sequence: BlazePoseSequence
                a BlazePoseSequence object

        Return
        -------
            data: dict
                A dict containing the sequence name and its serialized frame data
        """
        frame_data = [
            BlazePoseFrameSerializer.serialize(frame)
            for frame in blaze_pose_sequence.frames
        ]

        return {
            "name": blaze_pose_sequence.name,
            "frames": frame_data
        }

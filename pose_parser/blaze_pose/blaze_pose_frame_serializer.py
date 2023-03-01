from pose_parser.blaze_pose.blaze_pose_frame import BlazePoseFrame

class BlazePoseFrameSerializer:
    @staticmethod
    def serialize(blaze_pose_frame: BlazePoseFrame):
        """
        This method is responsible for returning a formatted data object
        for the passed blaze pose frame

        Parameters
        ---------
            blaze_pose_frame: BlazePoseFrame
                a BlazePoseFrame object
        
        Return
        -------
            data: dict
                A dict containing the sequence name and its serialized frame data 
        """
        return {}
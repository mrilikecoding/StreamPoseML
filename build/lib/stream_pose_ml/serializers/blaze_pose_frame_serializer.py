from stream_pose_ml.blaze_pose.blaze_pose_frame import BlazePoseFrame
from stream_pose_ml.serializers.angle_serializer import AngleSerializer
from stream_pose_ml.serializers.distance_serializer import DistanceSerializer
from stream_pose_ml.serializers.joint_serializer import JointSerializer


class BlazePoseFrameSerializer:
    @staticmethod
    def serialize(blaze_pose_frame: BlazePoseFrame):
        """This method is responsible for returning a formatted data object
        for the passed blaze pose frame.

        Args:
            blaze_pose_frame: BlazePoseFrame
                a BlazePoseFrame object

        Returns:
            data: dict
                A dict containing the sequence name and its serialized frame data
        """
        return {
            "type": "BlasePoseFrame",
            "sequence_id": blaze_pose_frame.sequence_id,
            "sequence_source": blaze_pose_frame.sequence_source,
            "frame_number": blaze_pose_frame.frame_number,
            "image_dimensions": blaze_pose_frame.image_dimensions,
            "has_joint_positions": blaze_pose_frame.has_joint_positions,
            "joint_positions": {
                joint_name: JointSerializer().serialize(joint_object)
                for (joint_name, joint_object) in blaze_pose_frame.joints.items()
            }
            if blaze_pose_frame.has_joint_positions
            else {},
            "angles": {
                angle_name: AngleSerializer().serialize(angle_object)
                for (angle_name, angle_object) in blaze_pose_frame.angles.items()
            }
            if blaze_pose_frame.has_joint_positions
            else {},
            "distances": {
                distance_name: DistanceSerializer().serialize(distance_object)
                for (
                    distance_name,
                    distance_object,
                ) in blaze_pose_frame.distances.items()
            }
            if blaze_pose_frame.has_joint_positions
            else {},
        }

from stream_pose_ml.geometry.angle import Angle
from stream_pose_ml.serializers.vector_serialzier import VectorSerializer


class AngleSerializer:
    @staticmethod
    def serialize(angle: Angle):
        return {
            "type": "Angle",
            "vector_1": angle.vector_1.name,
            "vector_2": angle.vector_2.name,
            "name": angle.name,
            "angle_2d": angle.angle_2d,
            "angle_2d_degrees": angle.angle_2d_degrees,
            "angle_3d": angle.angle_3d,
            "angle_3d_degrees": angle.angle_3d_degrees,
        }

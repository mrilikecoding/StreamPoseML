from pose_parser.geometry.angle import Angle
from pose_parser.serializers.vector_serialzier import VectorSerializer


class AngleSerializer:
    @staticmethod
    def serialize(angle: Angle):
        return {
            "type": "Angle",
            "vector_1": VectorSerializer().serialize(angle.vector_1),
            "vector_2": VectorSerializer().serialize(angle.vector_2),
            "name": angle.name,
            "angle_2d": angle.angle_2d,
            "angle_2d_radians": angle.angle_2d_degrees,
            "angle_3d": angle.angle_3d,
            "angle_3d_radians": angle.angle_3d_degrees,
        }

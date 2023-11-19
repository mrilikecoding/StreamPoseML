from stream_pose_ml.geometry.vector import Vector


class VectorSerializer:
    @staticmethod
    def serialize(vector: Vector):
        """
        This method is reponsible for formatting data from the Vector object
        """
        return {
            "type": "Vector",
            "name": vector.name,
            "joint_1_name": vector.joint_1.name,
            "joint_2_name": vector.joint_2.name,
            "direction_2d": vector.direction_2d,
            "direction_3d": vector.direction_2d,
            "direction_reverse_2d": vector.direction_reverse_2d,
            "direction_reverse_3d": vector.direction_reverse_2d,
            "x1": vector.x1,
            "x2": vector.x2,
            "y1": vector.y1,
            "y2": vector.y2,
            "z1": vector.z1,
            "z2": vector.z2,
            "x1_normalized": vector.x1_normalized,
            "x2_normalized": vector.x2_normalized,
            "y1_normalized": vector.y1_normalized,
            "y2_normalized": vector.y2_normalized,
            "z1_normalized": vector.z1_normalized,
            "z2_normalized": vector.z2_normalized,
        }

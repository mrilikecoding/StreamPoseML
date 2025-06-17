from stream_pose_ml.geometry.distance import Distance


class DistanceSerializer:
    @staticmethod
    def serialize(distance: Distance):
        return {
            "type": "Distance",
            "name": distance.name,
            "joint_name": distance.joint.name,
            "vector_name": distance.vector.name,
            "distance_2d": distance.distance_2d,
            "distance_3d": distance.distance_3d,
            "distance_2d_normalized": distance.distance_2d_normalized,
            "distance_3d_normalized": distance.distance_3d_normalized,
        }

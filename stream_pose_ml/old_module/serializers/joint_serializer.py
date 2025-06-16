from stream_pose_ml.geometry.joint import Joint


class JointSerializer:
    @staticmethod
    def serialize(joint: Joint):
        """
        This method is reponsible for formatting data from the Joint object
        """
        return {
            "type": "Joint",
            "name": joint.name,
            "image_dimensions": joint.image_dimensions,
            "x": joint.x,
            "y": joint.y,
            "z": joint.z,
            "x_normalized": joint.x_normalized,
            "y_normalized": joint.y_normalized,
            "z_normalized": joint.z_normalized,
        }

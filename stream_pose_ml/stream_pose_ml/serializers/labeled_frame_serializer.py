class LabeledFrameSerializer:
    """This class is responsible for serializing frame data for a dataset."""

    include_angles: bool
    include_distances: bool
    include_joints: bool

    def __init__(
        self,
        include_angles: bool = True,
        include_distances: bool = True,
        include_joints: bool = False,
        include_normalized: bool = True,
        include_z_axis: bool = False,
    ) -> None:
        """Init a LabeledFrameSerializer with options.
        Args:
            include_angles: bool
                whether to include angle data in this frame output
            include_distances: bool
                whether to include distance data in this frame output
            include_joints: bool
                whether to include joint data in this frame output

            TODO options for 3d / normalized
            TODO how to handle distance normalization - here or elsewhere?
        """
        self.include_angles = include_angles
        self.include_distances = include_distances
        self.include_joints = include_joints
        self.include_normalized = include_normalized
        self.include_z_axis = include_z_axis

    def serialize(self, frame: dict):
        """
        TODO - get the white list from ClipSerializer

        """
        row = {
            "video_id": frame["video_id"],
            "step_frame_id": 0
            if (frame["weight_transfer_type"] and frame["step_type"])
            else "NULL",
            "frame_number": frame["data"]["frame_number"],
            "weight_transfer_type": frame["weight_transfer_type"]
            if frame["weight_transfer_type"]
            else "NULL",
            "step_type": frame["step_type"] if frame["step_type"] else "NULL",
        }

        if self.include_joints:
            joints = self.serialize_joints(
                frame["data"]["joint_positions"],
                include_z_axis=self.include_z_axis,
                include_normalized=self.include_normalized,
            )
            row["joints"] = joints
        if self.include_angles:
            angles = self.serialize_angles(frame["data"]["angles"])
            row["angles"] = angles
        if self.include_distances:
            distances = self.serialize_distances(
                distances=frame["data"]["distances"],
                include_normalized=self.include_normalized,
                include_z_axis=self.include_z_axis,
            )
            row["distances"] = distances

        return row

    @staticmethod
    def serialize_angles(angles: dict):
        """Return a formatted dictionary of angles given passed params

        TODO include options for 2d/3d/normalized

        Note: the data for angles is combined with a "." in the key name to be consistent with
        the way pandas handles hierarchical column names and nested dictionaries. This is a convention
        that will make it easier to translate between data serialized here and ultimately output to csv
        and then read back into a dataframe. TODO - make this a consistent format between the model
        builder, the csv I/O, the way the web app loads in the right columns

        Args:
            angles: dict
                a dictionary of computed angles
        """
        angle_dictionary = {}
        for angle, data in angles.items():
            angle_dictionary[f"{angle}.angle_2d_degrees"] = data["angle_2d_degrees"]
        return angle_dictionary

    @staticmethod
    def serialize_distances(
        distances: dict, include_z_axis: bool = False, include_normalized: bool = False
    ):
        """This method serializes a distances object

        Note: the data for distances is combined with a "." in the key name to be consistent with
        the way pandas handles hierarchical column names and nested dictionaries. This is a convention
        that will make it easier to translate between data serialized here and ultimately output to csv
        and then read back into a dataframe.

        Args:
            distances: dict
                a dictionary of raw distance data
            include_z_axis: bool
                whether to include z axis
            include_normalized: bool
                whether to include the normalized data in the dictionary

        Returns:
            distance_dictionary: dict
                data serialized according to input params
        """
        distance_dictionary = {}
        for distance, data in distances.items():
            distance_dictionary[f"{distance}.distance_2d"] = data["distance_2d"]
            if include_normalized:
                distance_dictionary[f"{distance}.distance_2d_normalized"] = data[
                    "distance_2d_normalized"
                ]
            if include_z_axis:
                distance_dictionary[f"{distance}.distance_3d"] = data["distance_3d"]
                if include_normalized:
                    distance_dictionary[f"{distance}.distance_3d_normalized"] = data[
                        "distance_3d_normalized"
                    ]

        return distance_dictionary

    @staticmethod
    def serialize_joints(
        joints, include_z_axis: bool = False, include_normalized: bool = False
    ):
        """This method serializes a joint positions object

        Note: the data for joints is combined with a "." in the key name to be consistent with
        the way pandas handles hierarchical column names and nested dictionaries. This is a convention
        that will make it easier to translate between data serialized here and ultimately output to csv
        and then read back into a dataframe.

        Args:
            distances: dict
                a dictionary of raw joint position data
            include_z_axis: bool
                whether to include z axis
            include_normalized: bool
                whether to include the normalized data in the dictionary

        Returns:
            joints_dictionary: dict
                data serialized according to input params
        """
        joints_dictionary = {}
        for joint, data in joints.items():
            joints_dictionary[f"{joint}.x"] = data["x"]
            joints_dictionary[f"{joint}.y"] = data["y"]
            if include_normalized:
                joints_dictionary[f"{joint}.x_normalized"] = data["x_normalized"]
                joints_dictionary[f"{joint}.y_normalized"] = data["y_normalized"]
            if include_z_axis:
                joints_dictionary[f"{joint}.z"] = data["z"]
                if include_normalized:
                    joints_dictionary[f"{joint}.z_normalized"] = data["z_normalized"]

        return joints_dictionary

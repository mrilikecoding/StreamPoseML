import copy


class DatasetOutputTransformerService:
    def __init__(self, opts) -> None:
        """
        Parameters
        -----------
            opts: dict[str, bool | int]
                specifications for this output
                e.g.

                opts = {
                    "include_joints": True,
                    "include_angles": True,
                    "include_distances": True,
                    "include_non_normalized_points": True,
                    "include_normalized_points": False,
                    "include_z_axis": False,
                    "decimal_precision": 2,
                    "pad_data": True, # make all example sequences the same length
                    "merge_labels": True, # merge all examples into one example list with labels included
                }
        """
        self.include_angles = opts["include_angles"]
        self.include_joints = opts["include_joints"]
        self.include_distances = opts["include_distances"]
        self.include_normalized_points = opts["include_normalized_points"]
        self.include_z_axis = opts["include_z_axis"]
        self.ndigits = opts["decimal_precision"]
        self.pad_data = opts["pad_data"]
        self.merge_labels = opts["merge_labels"]
        self.key_off_frames = opts["key_off_frames"]

    def format_dataset(self, generated_raw_dataset: dict) -> dict:
        """
        This method is responsible for taking a raw dataset generated from VideoDataloopMergeService
        and outputing in specified formats

        Parameters
        --------
            generated_raw_dataset: dict
                output from VideoDataloopMergeService.generate_dataset_from_map

        """

        formatted_data = {}
        labels = [label for label in generated_raw_dataset]
        for label in labels:
            formatted_data[label] = {"examples": [], "max_frame_count": 0}
            label_data = generated_raw_dataset[label]
            examples = [example for example in label_data]
            for example in examples:
                sequence = []
                frames = [frame for frame in example]
                frame_counter = 1
                for frame in frames:
                    frame_data = {}
                    frame_data["label"] = label
                    frame_data["frame_count"] = frame_counter
                    if self.include_joints:
                        joint_data = self.format_joint_data(frame["joint_positions"])
                        frame_data["joints"] = joint_data
                    if self.include_angles:
                        angle_data = self.format_angle_data(frame["angles"])
                        frame_data["angles"] = angle_data
                    if self.include_distances:
                        distance_data = self.format_distance_data(frame["distances"])
                        frame_data["distances"] = distance_data
                    sequence.append(frame_data)
                    frame_counter += 1
                    # this is for figuring out how much to pad the time series
                    if frame_counter > formatted_data[label]["max_frame_count"]:
                        formatted_data[label]["max_frame_count"] = frame_counter
                formatted_data[label]["examples"].append(sequence)
        if self.pad_data:
            formatted_data = self.pad_formatted_data(formatted_data)
        if self.merge_labels:
            formatted_data = self.merge_formatted_data_labels(formatted_data)
        return formatted_data

    def format_frame_keys(self, formatted_data):
        pass

    def replace_values_with_zeros(self, obj):
        for k, v in obj.items():
            if isinstance(v, dict):
                obj[k] = self.replace_values_with_zeros(v)
            elif isinstance(v, float):
                obj[k] = 0.0
        return obj

    def merge_formatted_data_labels(self, formatted_data):
        max_frame_count = max(
            [formatted_data[label]["max_frame_count"] for label in formatted_data]
        )
        formatted_data = self.pad_formatted_data(
            formatted_data=formatted_data, pad_to=max_frame_count
        )
        examples = []
        labels = []
        for label in formatted_data:
            labels.append(label)
            for example in formatted_data[label]["examples"]:
                examples.append(example)

        if self.key_off_frames:
            flattened_examples = []
            for example in examples:
                for frame in example:
                    flattened_examples.append(frame)

            examples = flattened_examples

        return {"labels": labels, "examples": examples}

    def pad_formatted_data(self, formatted_data, pad_to: int = None):
        for label in formatted_data:
            examples = formatted_data[label]["examples"]
            if pad_to:
                max_frame_count = pad_to
            else:
                max_frame_count = formatted_data[label]["max_frame_count"]
            for example in examples:
                frame_count = len(example)
                frames_to_add = max_frame_count - frame_count
                pad_frame = self.replace_values_with_zeros(copy.deepcopy(example[0]))
                for _ in range(1, frames_to_add + 1):
                    pad_frame = copy.deepcopy(pad_frame)
                    example.append(pad_frame)

        return formatted_data

    def pad_example(self, frame):
        pass

    def format_joint_data(self, joint_data):
        formatted_joints = {}
        for key in joint_data:
            formatted_joints[key] = {}
            formatted_joints[key]["x"] = round(joint_data[key]["x"], self.ndigits)
            formatted_joints[key]["y"] = round(joint_data[key]["y"], self.ndigits)
            if self.include_z_axis:
                formatted_joints[key]["z"] = round(joint_data[key]["z"], self.ndigits)

            if self.include_normalized_points:
                formatted_joints[key]["x_normalized"] = round(
                    joint_data[key]["x_normalized"], self.ndigits
                )
                formatted_joints[key]["y_normalized"] = round(
                    joint_data[key]["y_normalized"], self.ndigits
                )
            if self.include_normalized_points and self.include_z_axis:
                formatted_joints[key]["z_normalized"] = round(
                    joint_data[key]["z_normalized"], self.ndigits
                )
        return formatted_joints

    def format_angle_data(self, angle_data):
        formatted_angles = {}
        for key in angle_data:
            formatted_angles[key] = {}
            formatted_angles[key]["angle_2d"] = round(
                angle_data[key]["angle_2d"], self.ndigits
            )
            if self.include_z_axis:
                formatted_angles[key]["angle_3d"] = round(
                    angle_data[key]["angle_3d"], self.ndigits
                )
        return formatted_angles

    def format_distance_data(self, distance_data):
        formatted_distances = {}
        for key in distance_data:
            formatted_distances[key] = {}
            formatted_distances[key]["distance_2d"] = round(
                distance_data[key]["distance_2d"], self.ndigits
            )
            if self.include_z_axis:
                formatted_distances[key]["distance_3d"] = round(
                    distance_data[key]["distance_3d"], self.ndigits
                )
            if self.include_normalized_points:
                formatted_distances[key]["distance_2d_normalized"] = round(
                    distance_data[key]["distance_2d_normalized"], self.ndigits
                )
            if self.include_normalized_points and self.include_z_axis:
                formatted_distances[key]["distance_3d_normalized"] = round(
                    distance_data[key]["distance_3d_normalized"], self.ndigits
                )
        return formatted_distances

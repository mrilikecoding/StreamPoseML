import pandas as pd


class DatasetOutputTransformerService:
    def __init__(self, opts) -> None:
        """
        Args:
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
        self.merge_labels = opts["merge_labels"]
        self.save_to_csv = opts["save_to_csv"]

    def format_dataset(self, generated_raw_dataset: dict) -> dict:
        """
        This method is responsible for taking a raw dataset generated from VideoDataloopMergeService
        and outputing in specified formats

        Args:
            generated_raw_dataset: dict
                output from VideoDataloopMergeService.generate_dataset_from_map

        Returns:
            formatted_dataset: dict
                Returns: a data object with a list of all labels and a list of labeled examples

                {
                    labels:
                        ['Right Step', 'Successful Weight Transfer', 'Left Step'],
                    'examples':
                        [{...}, ...]
                }
        """
        formatted_data = {}
        labels = [label for label in generated_raw_dataset]
        for label in labels:
            formatted_data[label] = {"examples": [], "max_frame_count": 0}
            label_data = generated_raw_dataset[label]
            examples = [example for example in label_data]
            for example in examples:
                example_data = {"label": label, "frames": {}}
                frames = [frame for frame in example]
                for frame_counter, frame in enumerate(frames):
                    frame_label = f"frame_{frame_counter + 1}"
                    example_data["frames"][frame_label] = {}
                    if self.include_joints:
                        joint_data = self.format_joint_data(frame["joint_positions"])
                        example_data["frames"][frame_label]["joints"] = joint_data
                    if self.include_angles:
                        angle_data = self.format_angle_data(frame["angles"])
                        example_data["frames"][frame_label]["angles"] = angle_data
                    if self.include_distances:
                        distance_data = self.format_distance_data(frame["distances"])
                        example_data["frames"][frame_label]["distances"] = distance_data
                    # this is for figuring out how much to pad the time series
                    if frame_counter > formatted_data[label]["max_frame_count"]:
                        formatted_data[label]["max_frame_count"] = frame_counter
                formatted_data[label]["examples"].append(example_data)
        if self.merge_labels:
            formatted_data = self.merge_formatted_data_labels(formatted_data)
        if self.save_to_csv:
            self.save_formatted_data_to_csv(formatted_data=formatted_data)

        return formatted_data

    def save_formatted_data_to_csv(self, formatted_data, filename="test.csv"):
        df = pd.json_normalize(formatted_data["examples"])
        df.fillna(0.0, inplace=True)
        df.to_csv(filename)

    def replace_values_with_zeros(self, obj):
        for k, v in obj.items():
            if isinstance(v, dict):
                obj[k] = self.replace_values_with_zeros(v)
            elif isinstance(v, float):
                obj[k] = 0.0
        return obj

    def merge_formatted_data_labels(self, formatted_data):
        examples = []
        labels = []
        for label in formatted_data:
            labels.append(label)
            for example in formatted_data[label]["examples"]:
                examples.append(example)

        return {"labels": labels, "examples": examples}

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

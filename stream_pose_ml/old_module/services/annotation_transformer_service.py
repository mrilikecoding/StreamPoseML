import yaml
import os


def get_nested_key(data: dict, key: str):
    """Access nested dictionary keys based on a dot-separated key string."""
    keys = key.split(".")
    for k in keys:
        data = data[k]
    return data


def find_project_root(identifying_file="config.yml"):
    """Find the root directory of the project by looking for an identifying file."""
    current_path = os.path.abspath(os.curdir)

    while True:
        files_in_current_path = os.listdir(current_path)

        if identifying_file in files_in_current_path:
            return current_path

        # Move up one directory level
        parent_path = os.path.dirname(current_path)

        # If we've already reached the root directory, stop
        if parent_path == current_path:
            raise Exception(f"Root directory with {identifying_file} not found!")

        current_path = parent_path


class AnnotationTransformerService:
    """
    This class is responsible for marrying video frame data and annotations.
    """

    @staticmethod
    def load_annotation_schema(schema_filename: str = "config.yml") -> dict:
        """Loads the annotation schema from a YAML file."""
        project_root = find_project_root()
        schema_path = os.path.join(project_root, schema_filename)

        with open(schema_path, "r") as ymlfile:
            return yaml.load(ymlfile, Loader=yaml.FullLoader)["annotation_schema"]

    @staticmethod
    def update_video_data_with_annotations(
        annotation_data: dict, video_data: dict, schema: dict = None
    ) -> tuple:
        """Merged video and annotation data.

        This method accepts a dictionary of annotation_data and a serialized video_data dictionary
        and then extracts the corresponding clip from the video frame data and stores it with the right
        annotation label.

        Args:
            annotation_data: dict
                Raw json annotation data matching defined schema corresponding to the passed video data
            video_data: dict
                Serialized video data for each frame
            schema: dict
                Annotation schema specifying the structure of annotation_data
        Returns:
            frame_lists: tuple[list, list, list]
                returns a tuple of lists of all_frames, labeled_frames, unlabeled_frames
        """
        if schema is None:
            schema = AnnotationTransformerService.load_annotation_schema()

        # Extract annotation information based on provided schema
        annotations_key = schema["annotations_key"]
        clip_annotation_map = [
            {
                "label": get_nested_key(
                    annotation, schema["annotation_fields"]["label"]
                ),
                "frame": get_nested_key(
                    annotation, schema["annotation_fields"]["start_frame"]
                ),
                "endFrame": get_nested_key(
                    annotation, schema["annotation_fields"]["end_frame"]
                ),
            }
            for annotation in annotation_data[annotations_key]
        ]

        label_hierarchy = schema["label_class_mapping"]

        # Determine top level column names
        label_columns = set(label_hierarchy.values())
        labeled_frames = []
        unlabeled_frames = []
        all_frames = []

        video_name = video_data["name"]
        for frame, frame_data in video_data["frames"].items():
            data = {column: None for column in label_columns}
            # For each frame, assign top level column values to their appropriate labels
            for annotation in clip_annotation_map:
                label_column = label_hierarchy[annotation["label"]]
                start_frame = annotation["frame"]
                end_frame = annotation["endFrame"]
                if start_frame <= frame_data["frame_number"] <= end_frame:
                    data[label_column] = annotation["label"]
            data["data"] = frame_data
            data["video_id"] = video_name

            all_frames.append(data)
            if all([data[column] for column in label_columns]):
                labeled_frames.append(data)
            else:
                unlabeled_frames.append(data)

        return (all_frames, labeled_frames, unlabeled_frames)


class AnnotationTransformerServiceError(Exception):
    """Raised when there's a problem in the AnnotationTransformerService"""

    pass

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[
    3
]  # /Users/nathangreen/Development/stream_pose_ml
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from unittest.mock import mock_open, patch

from stream_pose_ml.utils.path_utility import (
    get_base_path,
    get_file_name,
    get_file_paths_in_directory,
    write_to_json_file,
)


class TestGetFilePathsInDirectory:
    """Test the get_file_paths_in_directory function."""

    @patch("stream_pose_ml.utils.path_utility.glob.iglob")
    def test_get_file_paths_no_extension(self, mock_iglob):
        """Test getting file paths without specifying an extension."""
        # Setup
        mock_iglob.return_value = [
            "/path/to/file1.txt",
            "/path/to/file2.jpg",
            "/path/to/subdirectory/file3.pdf",
        ]

        # Execute
        result = get_file_paths_in_directory("/path/to")

        # Verify
        mock_iglob.assert_called_once_with("/path/to/**/*", recursive=True)
        assert len(result) == 3
        assert "/path/to/file1.txt" in result
        assert "/path/to/file2.jpg" in result
        assert "/path/to/subdirectory/file3.pdf" in result

    @patch("stream_pose_ml.utils.path_utility.glob.iglob")
    def test_get_file_paths_with_str_extension(self, mock_iglob):
        """Test getting file paths by specifying a string extension."""
        # Setup
        mock_iglob.return_value = [
            "/path/to/file1.txt",
            "/path/to/file2.txt",
            "/path/to/subdirectory/file3.txt",
        ]

        # Execute
        result = get_file_paths_in_directory("/path/to", extension="txt")

        # Verify
        mock_iglob.assert_called_once_with("/path/to/**/*.txt", recursive=True)
        assert len(result) == 3
        assert all(file.endswith(".txt") for file in result)

    @patch("stream_pose_ml.utils.path_utility.glob.iglob")
    def test_get_file_paths_with_list_extension(self, mock_iglob):
        """Test getting file paths by specifying a list of extensions."""
        # Setup
        mock_iglob.side_effect = [
            ["/path/to/file1.jpg", "/path/to/file2.jpg"],
            ["/path/to/file3.png", "/path/to/file4.png"],
        ]

        # Execute
        result = get_file_paths_in_directory("/path/to", extension=["jpg", "png"])

        # Verify
        assert mock_iglob.call_count == 2
        mock_iglob.assert_any_call("/path/to/**/*.jpg", recursive=True)
        mock_iglob.assert_any_call("/path/to/**/*.png", recursive=True)
        assert len(result) == 4
        assert "/path/to/file1.jpg" in result
        assert "/path/to/file2.jpg" in result
        assert "/path/to/file3.png" in result
        assert "/path/to/file4.png" in result


class TestGetBasePath:
    """Test the get_base_path function."""

    def test_get_base_path_with_file(self):
        """Test getting the base path from a file path."""
        # Setup
        file_path = "/path/to/directory/file.txt"

        # Execute
        result = get_base_path(file_path)

        # Verify
        assert result == "/path/to/directory"

    def test_get_base_path_with_directory(self):
        """Test getting the base path from a directory path."""
        # Setup
        file_path = "/path/to/directory/"

        # Execute
        result = get_base_path(file_path)

        # Verify
        assert result == "/path/to/directory"

    def test_get_base_path_with_root_file(self):
        """Test getting the base path from a file in the root directory."""
        # Setup
        file_path = "/file.txt"

        # Execute
        result = get_base_path(file_path)

        # Verify
        assert result == "/"


class TestGetFileName:
    """Test the get_file_name function."""

    def test_get_file_name_with_extension(self):
        """Test getting the file name with extension."""
        # Setup
        file_path = "/path/to/directory/file.txt"

        # Execute
        result = get_file_name(file_path)

        # Verify
        assert result == "file.txt"

    def test_get_file_name_without_extension(self):
        """Test getting the file name without extension."""
        # Setup
        file_path = "/path/to/directory/file.txt"

        # Execute
        result = get_file_name(file_path, omit_extension=True)

        # Verify
        assert result == "file"

    def test_get_file_name_for_hidden_file(self):
        """Test getting the name of a hidden file."""
        # Setup
        file_path = "/path/to/directory/.hidden_file"

        # Execute
        result = get_file_name(file_path)

        # Verify
        assert result == ".hidden_file"

        result_without_ext = get_file_name(file_path, omit_extension=True)
        assert result_without_ext == ".hidden_file"

    def test_get_file_name_for_file_without_extension(self):
        """Test getting the name of a file without an extension."""
        # Setup
        file_path = "/path/to/directory/README"

        # Execute
        result = get_file_name(file_path)

        # Verify
        assert result == "README"

        result_without_ext = get_file_name(file_path, omit_extension=True)
        assert result_without_ext == "README"


class TestWriteToJsonFile:
    """Test the write_to_json_file function."""

    @patch("stream_pose_ml.utils.path_utility.os.makedirs")
    @patch("stream_pose_ml.utils.path_utility.json.dumps")
    @patch("builtins.open", new_callable=mock_open)
    @patch("builtins.print")
    def test_write_to_json_file_with_dict(
        self, mock_print, mock_file, mock_dumps, mock_makedirs
    ):
        """Test writing a dictionary to a JSON file."""
        # Setup
        file_path = "/path/to/directory"
        file_name = "data.json"
        data = {"key": "value", "number": 42}

        mock_dumps.return_value = '{\n    "key": "value",\n    "number": 42\n}'

        # Execute
        result = write_to_json_file(file_path, file_name, data)

        # Verify
        mock_makedirs.assert_called_once_with(file_path, exist_ok=True)
        mock_dumps.assert_called_once_with(data, indent=4)
        mock_file.assert_called_once_with(f"{file_path}/{file_name}", "w")
        mock_file().write.assert_called_once_with(
            '{\n    "key": "value",\n    "number": 42\n}'
        )
        mock_print.assert_called_once_with(
            f"Successfully wrote {file_path}/{file_name}."
        )
        assert result is True

    @patch("stream_pose_ml.utils.path_utility.os.makedirs")
    @patch("stream_pose_ml.utils.path_utility.json.dumps")
    @patch("builtins.open", new_callable=mock_open)
    @patch("builtins.print")
    def test_write_to_json_file_with_list(
        self, mock_print, mock_file, mock_dumps, mock_makedirs
    ):
        """Test writing a list to a JSON file."""
        # Setup
        file_path = "/path/to/directory"
        file_name = "data.json"
        data = [{"id": 1, "name": "item1"}, {"id": 2, "name": "item2"}]

        mock_dumps.return_value = (
            '[\n    {"id": 1, "name": "item1"},\n    {"id": 2, "name": "item2"}\n]'
        )

        # Execute
        result = write_to_json_file(file_path, file_name, data)

        # Verify
        mock_makedirs.assert_called_once_with(file_path, exist_ok=True)
        mock_dumps.assert_called_once_with(data, indent=4)
        mock_file.assert_called_once_with(f"{file_path}/{file_name}", "w")
        mock_file().write.assert_called_once_with(
            '[\n    {"id": 1, "name": "item1"},\n    {"id": 2, "name": "item2"}\n]'
        )
        mock_print.assert_called_once_with(
            f"Successfully wrote {file_path}/{file_name}."
        )
        assert result is True

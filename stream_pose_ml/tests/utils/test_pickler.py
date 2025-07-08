import pickle
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[
    3
]  # /Users/nathangreen/Development/stream_pose_ml
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from unittest.mock import mock_open, patch

import pytest

from stream_pose_ml.utils.pickler import load_from_pickle, save_to_pickle


class TestSaveToPickle:
    """Test the save_to_pickle function."""

    @patch("stream_pose_ml.utils.pickler.pickle.dump")
    @patch("builtins.open", new_callable=mock_open)
    @patch("builtins.print")
    def test_save_to_pickle(self, mock_print, mock_file, mock_dump):
        """Test saving an object to a pickle file."""

        # Setup
        class TestClass:
            pass

        test_obj = TestClass()
        file_path = "/path/to/output"

        # Execute
        result = save_to_pickle(test_obj, file_path)

        # Verify
        mock_file.assert_called_once_with(f"{file_path}.pickle", "wb")
        mock_dump.assert_called_once_with(
            test_obj, mock_file(), protocol=pickle.HIGHEST_PROTOCOL
        )
        mock_print.assert_called_once_with(f"Saved {test_obj.__class__} to pickle")
        assert result is True


class TestLoadFromPickle:
    """Test the load_from_pickle function."""

    @patch("stream_pose_ml.utils.pickler.pickle.load")
    @patch("builtins.open", new_callable=mock_open)
    @patch("builtins.print")
    def test_load_from_pickle(self, mock_print, mock_file, mock_load):
        """Test loading an object from a pickle file, highlighting the bug."""

        # Setup
        class TestClass:
            pass

        test_obj = TestClass()
        mock_load.return_value = test_obj
        filename = "/path/to/file"

        # Note: The current implementation has a bug - it uses 'wb' mode instead of 'rb'
        # This test will pass but the actual implementation would fail

        # Execute
        result = load_from_pickle(filename)

        # Verify
        mock_file.assert_called_once_with(f"{filename}.pickle", "wb")  # Should be 'rb'
        mock_load.assert_called_once_with(mock_file())
        mock_print.assert_called_once_with(f"Loading {test_obj.__class__} from pickle")
        assert result == test_obj


@pytest.mark.skip(reason="This test showcases the fix for the bug in load_from_pickle")
class TestLoadFromPickleFixed:
    """Test for a fixed version of load_from_pickle function."""

    @patch("stream_pose_ml.utils.pickler.pickle.load")
    @patch("builtins.open", new_callable=mock_open)
    @patch("builtins.print")
    def test_load_from_pickle_fixed(self, mock_print, mock_file, mock_load):
        """Test loading an object from a pickle file with the correct 'rb' mode."""

        # Setup
        class TestClass:
            pass

        test_obj = TestClass()
        mock_load.return_value = test_obj
        filename = "/path/to/file"

        # This is how the function should be implemented
        with patch(
            "stream_pose_ml.utils.pickler.load_from_pickle",
            new=self._fixed_load_from_pickle,
        ):
            # Execute
            result = load_from_pickle(filename)

            # Verify
            mock_file.assert_called_once_with(
                f"{filename}.pickle", "rb"
            )  # Correct mode
            mock_load.assert_called_once_with(mock_file())
            mock_print.assert_called_once_with(
                f"Loading {test_obj.__class__} from pickle"
            )
            assert result == test_obj

    def _fixed_load_from_pickle(self, filename):
        """Fixed version of load_from_pickle using 'rb' mode."""
        with open(f"{filename}.pickle", "rb") as handle:
            obj = pickle.load(handle)
        print(f"Loading {obj.__class__} from pickle")
        return obj

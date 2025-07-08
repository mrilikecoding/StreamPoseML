import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[
    3
]  # /Users/nathangreen/Development/stream_pose_ml
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from stream_pose_ml.utils.utils import round_nested_dict


class TestRoundNestedDict:
    """Test the round_nested_dict function."""

    def test_round_nested_dict_with_simple_dict(self):
        """Test rounding values in a simple dictionary."""
        # Setup
        input_dict = {
            "value1": 1.23456789,
            "value2": 2.98765432,
            "string_value": "not_a_number",
            "int_value": 42,
        }

        # Execute
        result = round_nested_dict(input_dict, precision=3)

        # Verify
        assert result["value1"] == 1.235  # Rounded up
        assert result["value2"] == 2.988  # Rounded up
        assert result["string_value"] == "not_a_number"  # Unchanged
        assert result["int_value"] == 42  # Unchanged

    def test_round_nested_dict_with_nested_dict(self):
        """Test rounding values in a nested dictionary."""
        # Setup
        input_dict = {
            "outer_value": 3.14159265359,
            "nested": {
                "inner_value1": 2.71828182846,
                "inner_value2": 1.61803398875,
                "deeper": {"deepest_value": 0.57721566490},
            },
        }

        # Execute
        result = round_nested_dict(input_dict, precision=2)

        # Verify
        assert result["outer_value"] == 3.14
        assert result["nested"]["inner_value1"] == 2.72
        assert result["nested"]["inner_value2"] == 1.62
        assert result["nested"]["deeper"]["deepest_value"] == 0.58

    def test_round_nested_dict_with_list_values(self):
        """Test that the function properly handles list values (by not modifying
        them)."""
        # Setup
        input_dict = {
            "list_value": [1.23456, 2.34567, 3.45678],
            "regular_value": 4.56789,
        }

        # Execute
        result = round_nested_dict(input_dict, precision=3)

        # Verify
        assert result["list_value"] == [
            1.23456,
            2.34567,
            3.45678,
        ]  # Lists aren't processed
        assert result["regular_value"] == 4.568

    def test_round_nested_dict_default_precision(self):
        """Test rounding with the default precision of 4."""
        # Setup
        input_dict = {"value": 1.23456789}

        # Execute
        result = round_nested_dict(input_dict)  # Default precision=4

        # Verify
        assert result["value"] == 1.2346

    def test_round_nested_dict_with_zero_precision(self):
        """Test rounding to integer values (precision=0)."""
        # Setup
        input_dict = {"value1": 1.6, "value2": 2.3}

        # Execute
        result = round_nested_dict(input_dict, precision=0)

        # Verify
        assert result["value1"] == 2  # Rounded up
        assert result["value2"] == 2  # Rounded up

    def test_round_nested_dict_with_non_dict_input(self):
        """Test that the function correctly handles non-dictionary inputs."""
        # Setup & Execute
        float_result = round_nested_dict(3.14159, precision=2)
        int_result = round_nested_dict(42, precision=2)
        str_result = round_nested_dict("string", precision=2)
        list_result = round_nested_dict([1.2345, 2.3456], precision=2)

        # Verify
        assert float_result == 3.14
        assert int_result == 42  # Unchanged
        assert str_result == "string"  # Unchanged
        assert list_result == [1.2345, 2.3456]  # Unchanged

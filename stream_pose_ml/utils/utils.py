def round_nested_dict(item: dict, precision: int = 4):
    """This method takes a dictionary and recursively rounds float values to the indicated precision

    Args:
        item: dict
            A dictionary with nested keys and floats that need to be rounded
        precision: int
            How many decimals to round to
    """
    if isinstance(item, dict):
        return type(item)(
            (key, round_nested_dict(value, precision)) for key, value in item.items()
        )
    if isinstance(item, float):
        return round(item, precision)
    return item

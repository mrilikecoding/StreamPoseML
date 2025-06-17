""" This file has functions useful for taking a 
list of dictionaries of feature data (float values)
and computing a single dictionary where each key 
represents some temporal features of the feature's sequence
"""


def compute_standard_deviation(dict_list: list[dict]) -> dict:
    """
    Computes standard deviation pooling of a sequence of feature dictionaries.

    Args:
        features: A list of dictionaries representing a sequence of features,
                where each dictionary has the same keys.

    Returns:
        A dictionary representing the std-pooled features over the sequence.
    """
    std_dict = {}
    for key in dict_list[0].keys():
        mean = sum(d[key] for d in dict_list) / len(dict_list)
        std_dict[key] = sum((d[key] - mean) ** 2 for d in dict_list) / len(dict_list)

    return std_dict


def compute_max(dict_list: list[dict]) -> dict:
    """
    Computes max pooling of a sequence of feature dictionaries.

    Args:
        features: A list of dictionaries representing a sequence of features,
                where each dictionary has the same keys.

    Returns:
        A dictionary representing the max-pooled features over the sequence.
    """
    # Extract the keys from the first dictionary in the sequence
    keys = list(dict_list[0].keys())

    # Initialize the max-pooled features dictionary with zeros
    max_pooled = {key: 0 for key in keys}

    # Loop over the sequence of feature dictionaries and update the max-pooled values
    for feature in dict_list:
        for key in keys:
            max_pooled[key] = max(max_pooled[key], feature[key])

    return max_pooled


def compute_sum(dict_list: list[dict]) -> dict:
    """
    Computes sum pooling of a sequence of feature dictionaries.

    Args:
        features: A list of dictionaries representing a sequence of features,
                where each dictionary has the same keys.

    Returns:
        A dictionary representing the sum-pooled features over the sequence.
    """
    # Extract the keys from the first dictionary in the sequence
    keys = list(dict_list[0].keys())

    # Initialize the sum-pooled features dictionary with zeros
    sum_pooled = {key: 0 for key in keys}

    # Loop over the sequence of feature dictionaries and accumulate the values
    for feature in dict_list:
        for key in keys:
            sum_pooled[key] += feature[key]

    for key in sum_pooled:
        sum_pooled[key] = sum_pooled[key]

    return sum_pooled


def compute_average_value(dict_list: list[dict]) -> dict:
    """
    Computes average pooling of a sequence of feature dictionaries.

    Args:
        features: A list of dictionaries representing a sequence of features,
                where each dictionary has the same keys.

    Returns:
        A dictionary representing the mean-pooled features over the sequence.
    """
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict

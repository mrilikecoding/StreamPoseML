import pickle

# TODO this ain't working yet


def save_to_pickle(obj, file_path):
    """
    Saved passed object into pickle file at passed file path
    """
    with open(f"{file_path}.pickle", "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {obj.__class__} to pickle")
    return True


def load_from_pickle(filename):
    """
    Load pickle file at passed file path
    """
    with open(f"{filename}.pickle", "wb") as handle:
        obj = pickle.load(handle)
    print(f"Loading {obj.__class__} from pickle")
    return obj

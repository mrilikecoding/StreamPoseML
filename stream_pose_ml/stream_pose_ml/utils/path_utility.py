import os
import glob
import json
from pathlib import Path


def get_file_paths_in_directory(
    directory: str, extension: str | list = None
) -> list[str]:
    """Recursively get a list of file paths in a directory.

    Args:
        directory: str
            the directory to search
        extension: str | list
            if these files should all share an extension, these will only be returned
    Returns:
        file_paths: list[str]
            a list of file paths
    """
    if extension:
        if isinstance(extension, str):
            file_paths = [
                file_path
                for file_path in glob.iglob(
                    directory + f"/**/*.{extension}", recursive=True
                )
            ]
        elif isinstance(extension, list):
            file_paths = []
            for ext in extension:
                files = [
                    file_path
                    for file_path in glob.iglob(
                        directory + f"/**/*.{ext}", recursive=True
                    )
                ]
                file_paths += files

    else:
        file_paths = [
            file_path for file_path in glob.iglob(directory + f"/**/*", recursive=True)
        ]

    return file_paths


def get_base_path(file_path: str) -> str:
    """Get the base path without the file name.

    Args:
        file_path: str
            a path to a file
    Returns:
        path_to_file: str
            the path to a file without the filename
    """
    return os.path.split(file_path)[0]


def get_file_name(file_path: str, omit_extension: bool = False) -> str:
    """Get the filename including the extention without the base path.

    Args:
        file_path: str
            a path to a file
    Returns:
        filename: str
            the name of the file, including the extension
    """
    if omit_extension:
        return Path(file_path).stem
    return os.path.basename(file_path)


def write_to_json_file(file_path: str, file_name: str, data: list | dict):
    """Write the contents of a passed data object to a json file.

    Args:
        file_path: str
            where to put the file
        file_name: str
            what to name the file
        data: list | dict
            what to write to json

    Returns:
        success: bool
            True if we succeed

    """
    os.makedirs(file_path, exist_ok=True)

    json_file = json.dumps(data, indent=4)
    with open(f"{file_path}/{file_name}", "w") as f:
        f.write(json_file)
        print(f"Successfully wrote {file_path}/{file_name}.")

    return True

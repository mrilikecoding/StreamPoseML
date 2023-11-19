from setuptools import setup, find_packages

setup(
    name='StreamPoseML',
    version='0.1',
    packages=find_packages(where="pose_parser"),
    package_dir={"": "pose_parser"},
)

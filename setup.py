from setuptools import setup, find_packages

setup(
    name='StreamPoseML',
    version='0.1',
    packages=find_packages(where="stream_pose_ml"),
    package_dir={"": "stream_pose_ml"},
)

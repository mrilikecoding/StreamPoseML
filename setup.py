from setuptools import setup, find_packages
import os

def read_requirements():
    setup_py_dir = os.path.dirname(os.path.realpath(__file__))
    requirements_path = os.path.join(setup_py_dir, 'stream_pose_ml', 'requirements.txt')
    with open(requirements_path) as req:
        return req.read().splitlines()

setup(
    install_requires=read_requirements(),
    package_dir={'': 'stream_pose_ml'},
    packages=find_packages(where='stream_pose_ml'),
)


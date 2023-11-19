from setuptools import setup, find_packages

def read_requirements():
    with open('./stream_pose_ml/requirements.txt') as req:
        return req.read().splitlines()

setup(
    name='stream_pose_ml',
    version='0.1.0',  # Update the version number for new releases
    author='Nate Green',
    author_email='nate@nate.green',
    description='A toolkit for realtime video classification tasks.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mrilikecoding/StreamPoseML',
    project_urls={
        "Bug Tracker": "https://github.com/mrilikecoding/StreamPoseML/issues"
    },
    license='MIT', 
    package_dir={'': 'stream_pose_ml'},
    packages=find_packages(where='stream_pose_ml'),
    install_requires=read_requirements(),
    classifiers=[
        'Development Status :: 3 - Alpha',  # Update to '5 - Production/Stable' if applicable
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',  # Or your chosen license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.10',
)


# Poser

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Supported Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Windows%20%7C%20Linux-green)


Poser is an open-source, end-to-end toolkit for creating realtime, video-based classification experiments that rely on utilizing labeled data alongside captured body keypoint / pose data. The process for building a real-time video classification application typically looks something like this:

1. Collect video data
2. Label video data
3. Generate body keypoints from video
4. Compute features
5. Merge annotations/labels with keypoints/features into a dataset
6. Train a model
7. Run experiments
8. Deploy the trained the model
9. Classify real-time video captured via the web or some other input source
10. Actuate or send results outside the application

Poser aspires to help with steps 3-10, with the aim of making a system portable enough to be run wherever a Python environment can run in the case of steps 3-7, and wherever a Docker container can run, in the case of steps 8-10.

Additionally, Poser aims to provide flexibility with respect to coding and classification schemes. There are ready-baked video annotation + classification solutions out there, however they can be costly and not suited for every task. For a python dev/data-scientist, Poser provides a local laboratory for working with video data in a way that can mesh with your own workflow, on your own hardware, for free, and provides a starting point for creating your own portable real-time classification / actuation system. 


## Keypoint extraction

Poser currently uses Mediapipe to extract body keypoints. This is because Poser was developed to assist with realtime video classification tasks that could potentially run on relatively ubiquitous devices, perhaps in a therpeutic or live performance setting. The aim is to provide a system to enable anyone with a webcam to be able to classify video in real-time. 

You can certainly incorporate Poser into your own tooling. However, the best way to get started is to work within a Jupyter Notebook environment to bring everything together.

The process for extracting keypoints looks like this:

```
import pose_parser.jobs.process_videos_job as pv

pv.ProcessVideosJob().process_videos(
    src_videos_path='/path/to/source/videos',
    output_keypoints_data_path='/path/to/output/frame/keypoints',
    output_sequence_data_path='/path/to/output/video/sequences',
    write_keypoints_to_file=True,
    write_serialized_sequence_to_file=True,
    limit=None,
    configuration={}, # mediapipe configuration
    preprocess_video=True,
    return_output=False
)
```

You pass a directory containing your videos. Each video will be run through mediapipe. In the keypoints directory, namespaced to each video, json keypoint representations will be saved. Additionally, the entire video's keypoints will be serialized into a video sequence and stored in a parallel directory. These files can be used directly in a training regime, or you can use Poser's dataset building tools to format sequence data into other formats.

## Feature engineering

There are currently various options available that take the raw keypoint data and build upon it to generate normalized angle and distance measurements for use in building your dataset. These features are shown more in the examples below.

## Merging annotations with video keypoints / features

A pain point found in related research was the lack of accessible tooling for merging keypoint data from training videos with the actual labeled annotation data. While there are tools that exist to annotate videos for model training, often in research contexts a specific annotation process is used at perhaps a different than the training will occur, making it cumbersome to later merge the annotation data with the video data. This work can be tedious on top of the already tedious task of labeling the data to begin with. However this task is straightforward with Poser assuming you have structured annotation data. First, copy `config.example.yml` into `config.yml`.

Then update the annotation schema to match your annotation data. Poser assumes that you'll have one annotation file for each video you are training on and they can all live within one directory. However make sure they they share their name with the matching video. A single video may have many annotations. Currently Poser support JSON, but in future work other formats could be used. Your contribution to this area would be welcome!

Here's an example of a valid annotation file for video named `video1.mov`:

```
video1-annotation.json

 {
   "id": "63e10d737329c2fe92c8ae0a",
   "datasetId": "63bef4c53775a03d44271475",
   "metadata": {
     "system": {
       "ffmpeg": {
         "avg_frame_rate": "30000/1001",
         "width": 1920
       },
       "fps": 29.97002997002997,
     },
     "fps": 29.97002997002997,
     "startTime": 0.007
   },
   "name": "video1.webm",
   "annotations": [
     {
       "id": "63fe90715ff162c693fa0f3c",
       "datasetId": "63bef4c53775a03d44271475",
       "itemId": "63e10d737329c2fe92c8ae0a",
       "label": "Left Step",
       "metadata": {
         "system": {
           "startTime": 5.472133333333334,
           "endTime": 6.940266666666667,
           "frame": 164,
           "endFrame": 208,
           "openAnnotationVersion": "1.56.0-prod.31",
           "recipeId": "63bef4c5223e5c2a0a9e4227"
         },
         "user": {}
       },
       "source": "ui"
     },
     ...
   ],
   "annotationsCount": 3,
   "annotated": true
 }
```

Then here's what your `config.yml` should look like.

```
annotation_schema: # assume one annotation file per video where there is a list of annotations
  annotations_key: "annotations" # the key in the annotation file that contains the list of annotations
  annotation_fields: # the fields in the annotation file that map to the video data
    label: label # the label field in the annotation list
    start_frame: metadata.system.frame # the starting video frame for the annotation
    end_frame: metadata.system.endFrame # the ending video frame for the annotation
  label_class_mapping: # for each label (Key), map to a class (Value), i.e. Dog: animal, or Truck: vehicle, or 0: has_something
    Left Step: step_type
    Right Step: step_type
    Successful Weight Transfer: weight_transfer_type
    Failure Weight Transfer: weight_transfer_type
```

## Creating datasets with features

Poser was built while conducting studies of Parkinson's Disease patients in dance therapy settings. From these efforts, you can see several Jupyter notebook examples showing how to use Poser to built a training dataset.

To get a feel for building your dataset using Poser, see `/pose_parser/notebooks/dataset_for_ui.ipynb`

The process looks like this:

```
import pose_parser.jobs.build_and_format_dataset_job as data_builder 

# This is the main class that does all the work
db = data_builder.BuildAndFormatDatasetJob()

# Here you'll specift the path to you annotations and Poser generated sequences
dataset = db.build_dataset_from_data_files(
    annotations_data_directory=source_annotations_directory,
    sequence_data_directory=sequence_data_directory,
    limit=None,
)

formatted_dataset = db.format_dataset(
    dataset=dataset,
    pool_frame_data_by_clip=False,
    decimal_precision=4,
    include_unlabeled_data=True,
    include_angles=True,
    include_distances=True,
    include_normalized=True,
    segmentation_strategy="flatten_into_columns",
    segmentation_splitter_label="step_type",
    segmentation_window=10,
    segmentation_window_label="weight_transfer_type",
)

db.write_dataset_to_csv(
    csv_location=merged_annotation_output_directory,
    formatted_dataset=formatted_dataset,
    filename="preprocessed_flatten_on_example_10_frames_5"
)
```

For most training tasks you may not want to get too clever with the features and may just want to train on flat representations of raw keypoints. 

The simplest approach is:

```
formatted_dataset = db.format_dataset(
    dataset=dataset,
    pool_frame_data_by_clip=False,
    decimal_precision=4,
    include_unlabeled_data=True,
    include_angles=False,
    include_distances=False,
    include_normalized=False,
    segmentation_strategy="none",
)
```

This will give you one row per frame with columns for each x, y, z coordinate in addition to your labeled data. From there you can use pandas or whatever you like to further window or segment your data.


## Training models

There are several convenience methods abstracted into a Model Builder class created to speed up iterations and model evaluation. See the notebooks for usage examples. However, once you have your dataset you can use whatever process you like to train models.

## Saving your model

If you want to use your trained model in Poser's web application, you'll need to save it as a "pickle" so that it can be loaded into the application server at runtime. You may need to wrap it in a class before you do this such that when it is loaded it responds  with a result when the method "predict" is called on it.

## Running the Web Applictaion

First, as mentioned above, you'll need a trained classifier saved to a pickle file (in Python). The model should implement a "predict" method that takes an array of examples to classify. For realtime video classification generally you'll want to pass a single example

The pickle object should be shaped like this:

```
{
  classifier: <your_trained_model>
}
```

Place this pickle file in `data/trained_models`

Provided is a simple Flask API that sits behind a React UI. The UI was tailored for our specific use case in classifying types of steps captured via webcam, however you can adapt this for your own model classification scheme.

To run the app:

1. Visit docker.com and sign up for an account.
2. Download the Docker for Desktop client for your your sytem, launch, and log in.
3. Run `start.sh`

This should install the necessary dependencies and then launch the application in your default browser. 

4. When you're done, run `stop.sh` to gracefully end the application processes.

## Installation

1. Set up a python environment via conda or pyenv or other preferred tool.
2. Install via `requirements.txt` located in the `pose_parser` folder.

To run the web app, you'll want to do `docker-compose up`. The app should be available on `localhost:3000`. The API is served on `localhost:5001` and should be accessible from the web app.

### Debugging the web app / python API via VS Code within docker

Follow this guide if you're using VS Code and wish to debug the python API while running the containerized web app. 

This setup may be helpful within the `.vscode` folder:

In tasks.json:

```
{
    "version": "2.0.0",
    "tasks": [
      {
        "label": "Docker: Compose Up",
        "type": "shell",
        "command": "docker-compose up -d && sleep 15",
        "options": {
          "cwd": "${workspaceFolder}"
        },
        "group": {
          "kind": "build",
          "isDefault": true
        },
        "problemMatcher": []
      }
    ]
  }
  ```

  Note - the `sleep 15` there is to give docker-compose time to spin up everything, otherwise the debugger fails to attach. If you notice the debugger still failing to attach you may need to increase this time.

In `launch.json`:
```
    "configurations": [
        {
            "name": "Python: Remote Attach",
            "type": "python",
            "request": "attach",
            "port": 5678,
            "host": "localhost",
            "pathMappings": [
              {
                "localRoot": "${workspaceFolder}/pose_parser",
                "remoteRoot": "/usr/src/app"
              }
            ],
            "preLaunchTask": "Docker: Compose Up",
            "postDebugTask": "Docker: Compose Down"
          }
    ],
    "compounds": [
        {
          "name": "Docker Compose: Debug",
          "configurations": ["Python: Remote Attach"]
        }
      ]
}
```

Then, after running `docker-compose up`, run the VS debug process "Docker-Compose: Debug" for the app, and you should be able to set breakpoints / debug etc.

Note - within the flow of socketio, you won't be able to see changes reloaded automatically for various annoying reasons. So if you debug and make a change, you'll need to restart the debugger.

## Workflow

Locally I've been running experiments, training models, testing, and writing notebooks outside of docker. The purpose of the dockerized container is to facilitate a deployed ML Model accessbile via API from the React front end. Therefore if you are working on the back end and make environment dependency changes, you'll need to rebuild the docker container with the updated dependencies if you want to use the web app. This is done by following these steps:

1. Make sure your local isolated python environment is activated.
2. `cd pose_parser` (enter into the pose_parser directory)
3. `pip list --format=freeze > requirements.txt` -- generate the list of dependencies from your local environment. Note, this format is necessary because otherwise pip tends to create strange `file://` paths - we just want to specify package versions.
4. From the root directory run `docker-compose build pose_parser_api`. 

NOTE - you may need to futz with dependencies / versions from errors that are generated.

## Tests

To start run `python setup.py` to set paths correctly. Then simply:

`pytest`

Note: tests work for some of the modules but have fallen behind... contributions wanted! You'll note some failures etc. 

## Citing 

If you use this code in any research, please cite it so that others may benefit from knowing about this project. See [CITATION.txt](CITATION.txt).

## Contributions

Your contributions would be enthusiastically welcomed! While there's not a formal roadmap, there are plenty of TODOs and with more people using this, the hope is that a roadmap will emerge. See [CONTRIBUTING.md](CONTRIBUTING.md).

### TODOs
* fix error that sends keypoints to server when classification is turned off
* make UI nicer and create an easier template to adapt
* create a nicer scheme for integrating models and their data schema
  * this probably looks like an abstract class that can wrap models as well as multi-stage classifiers with a single predict method. This abstract class should also incorporate the data schema. There should be a utility that can go from skel data to the schema... this'll be a big feature.
* add citation file
* Finish testing modules - use test doubles
* Refactor object creation - some complicated init anti-patterns are here
* Build out an ETL wrapper and simplify job logic - there are some parameter sinkholes in the current flow
* Build a utility for looking at video and annotations
* Model builder methods are cumbersome and untested - need to split up model creation from grid search / random search logic
   

# Building Dockerfiles

You may with to build and push the Dockerfiles to your own registry to deploy an application based on Poser. There are two main components with respect to Poser's web application: the API and the UI. To build each:
```
cd pose_parser && docker build -t myuser/pose_parser_api:latest -f Dockerfile .
cd web_ui && docker build -t myuser/web_ui:latest -f Dockerfile .
```

Then you can push them and deploy them however you see fit, e.g. ECR / K8s.
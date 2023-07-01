# Poser

A realtime video classification web application and python toolkit for building and training models.

This system is meant to run locally on your machine. The web application ships within Docker containers. The model toolkits are best run within your own isolated Python environment, outside of Docker.

## Running the Web Applictaion

First, you'll need a trained classifier saved to a pickle file (in Python). The model should implement a "predict" method that takes an array of examples to classify. For realtime video classification generally you'll want to pass a single example

The pickle object should be shaped like this:

```
{
  classifier: <your_trained_model>
}
```

Place this pickle file in `data/trained_models`

Then to run the app:

1. Visit docker.com and sign up for an account.
2. Download the Docker for Desktop client for your your sytem, launch, and log in.
3. Run `start.sh`

This should install the necessary dependencies and then launch the application in your default browser. 

4. When you're done, run `stop.sh` to gracefully end the application processes.

## June 2023

## Work in progress


This project aims to simplify running keypoint extraction from Mediapipe (BlazePose), computing commonly used angle and distance measurements and modeling them in memory, and marrying the data with annotation data. Finally this project provides a model builder interface to simplify running machine learning experiments.

You'll also find the beginnings of a web application here - the intention is to eventually have a means to capture video in realtime, pipe the video data to the server layer (via RTC) and then through a pose parsing pipeline that can return the results of a classification.

This is a WIP and not ready for primetime - the classes and methods are well documented but there's no good usage documentation. However, the notebook examples should provide enough guidance for now.

## Notebooks

Notebooks can be found by navigating to `pose_parser/notebooks`

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

Tests work for some of the modules but have fallen behind... 
If you run them you'll see some failures. Additionally, tests are not well isolated just because before building notebooks I was building datasets by running through tests... ick. So.. I'd recommend avoiding the tests for right now.

## Citing 

If you use this, please cite me. But this is not ready for use. Please don't use this yet lol.

## TODOs
* add citation file
* Finish testing modules - use test doubles
* Refactor object creation - some complicated init anti-patterns are here
* Build out an ETL wrapper and simplify job logic - there are some parameter sinkholes in the current flow
* Settle on API and build out web app
* Build a utility for looking at video and annotations
* Abstract the annotation layer - we used Dataloop, but what we should do is define a way to load in an annotation schema so that other tools can be dropped in (annotation strategies with json schema)
* Model builder methods are cumbersome and untests - need to split up model creation from grid search / random search logic
   

# Building Dockerfiles

cd pose_parser && docker build -t myuser/pose_parser_api:latest -f Dockerfile .
cd web_ui && docker build -t mrilikecoding/web_ui:latest -f Dockerfile .
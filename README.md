# Pose Parser aka Poser

## Work in progress


This project aims to simplify running keypoint extraction from Mediapipe (BlazePose), computing commonly used angle and distance measurements and modeling them in memory, and marrying the data with annotation data. Finally this project provides a model builder interface to simplify running machine learning experiments.

You'll also find the beginnings of a web application here - the intention is to eventually have a means to capture video in realtime, pipe the video data to the server layer (via RTC) and then through a pose parsing pipeline that can return the results of a classification.

This is a WIP and not ready for primetime - the classes and methods are well documented but there's no good usage documentation. However, the notebook examples should provide enough guidance for now.

## Notebooks

Notebooks can be found by navigating to `pose_parser/notebooks`

## Installation

1. Set up a python environment via conda or pyenv or other preferred tool.
2. Install via `requirements.txt` located in the `pose_parser` folder.

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
   

  

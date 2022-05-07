# Attendance-Project

This Attendance Project will be used as an Mess In/Out system for my University's mess. 

Built using [dlib](http://dlib.net/)'s state-of-the-art face recognition
built with deep learning. The model has an accuracy of 99.38% on the
[Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) benchmark.

This can also be used as a `Standalone Application` by exporting the project as an executable. (The project has to be exported again in case of a new/modified training dataset)

I used the Convolutional Neural Network `CNN` instead of using HOG (Can be modified later if facing performance issues)

[![PyPI](https://img.shields.io/pypi/v/face_recognition.svg)](https://pypi.python.org/pypi/face_recognition)
[![Documentation Status](https://readthedocs.org/projects/face-recognition/badge/?version=latest)](http://face-recognition.readthedocs.io/en/latest/?badge=latest)



## Requirements:

Make sure you have the following modules installed.

`pip`
`cmake`
`numpy`
`dlib`
`opencv-python`
`face_recognition`

They can be installed via the following command using command line once you have installed pip:

```bash
pip3 install numpy dlib opencv-python face_recognition
```

## Creating a Standalone Executable
If you want to create a standalone executable that can run without the need to install `python` or `face_recognition`, you can use [PyInstaller](https://github.com/pyinstaller/pyinstaller).

## Tutorial:

Edit the path in `Attendance.py` 
```bash
path = 'Images'
```

Copy all the images required for the neural network into the `Images` folder and then run `Attendance.py`

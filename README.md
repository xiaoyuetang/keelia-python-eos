# keelia-python-eos
 
An implementation of modified-eos using python

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
dlib
cv2
numpy
modified-eos
```

### How to setup modified-eos

```
$ clone https://github.com/xiaoyuetang/eos.git
$ cd eos && python setup.py install
```

## Usage

**$ python generate_faces.py** to generate cropped faces with size 240
**$ python generate_pts.py** to generate landmarks for each cropped face
**$ python ground_truth.py** to generate .npy file storing dictionary of groud truth values:

{"tx": tx, "ty": ty, "scale": scale, "yaw": yaw, "roll": roll, "pitch": pitch, "pca_shape_coefficients": pca_shape_coefficients, "expression_coefficients": expression_coefficients}

## Build With
* [eos](https://github.com/patrikhuber/eos) - A lightweight 3D Morphable Face Model fitting library in modern C++14

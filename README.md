# Face Search VGG16

Implement Face Search using VGG16 feature value

# Requirement

Keras

Caffe

# Preparation

## Download Face Detection Pretrained Model

Converted from <https://github.com/dannyblueliu/YOLO-version-2-Face-detection>

<http://www.abars.biz/keras/face.prototxt>

<http://www.abars.biz/keras/face.caffemodel>

Download face.prototxt and face.caffemodel and put in the pretrain folder.

# Demo

First, please capture some faces.

Captured face placed in runtime/faces and runtime/feature

`python face_search.py capture`

Here is a run face seach.

`python face_search.py search`

Here is a run face seach from file.

`python face_search.py search input.jpg`

#  Dataset

## Use your own datset

Put face images in local/faces folder.

Here is a extract feature value from local/faces to local/feature.

`python feature_extract.py`

## Use adience benchmark dataset

Download AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification dataset (agegender folder)  and put in the dataset folder.

https://www.openu.ac.il/home/hassner/Adience/data.html#agegender

Here is a create local/faces.

`perl adience_benchmark_to_faces.pl`

Here is a extract feature value from local/faces to local/feature.

`python feature_extract.py`


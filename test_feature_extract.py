import os
import cv2
import numpy as np
import glob
import pickle
from datetime import datetime

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

# Read image features
features = []
img_paths = []
for feature_path in glob.glob("local/feature/*"):
    features.append(pickle.load(open(feature_path, 'rb')))
    img_paths.append('local/faces/' + os.path.splitext(os.path.basename(feature_path))[0] + '.jpg')

# Read models
base_model = VGG16(weights='imagenet')
keras_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
keras_model.summary()

# Search face
file = "landmark_aligned_face.1.9904044896_cb797f78d2_o.jpg"
img = cv2.imread('./local/faces/'+file)
img = cv2.resize(img, (224, 224))
img = img[...,::-1]  #BGR 2 RGB
data = np.array(img, dtype=np.float32)
data.shape = (1,) + data.shape
data -= 128
pred = keras_model.predict(data)[0]
feature = pred / np.linalg.norm(pred)  # Normalize

# Search result
query = feature
dists = np.linalg.norm(features - query, axis=1)  # Do search
ids = np.argsort(dists)[:30] # Top 30 results
scores = [(dists[id], img_paths[id]) for id in ids]

print scores

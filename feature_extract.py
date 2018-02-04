import os
os.environ['GLOG_minloglevel'] = '2' 

import cv2
import numpy as np
import pickle

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

files = os.listdir('./local/faces')

base_model = VGG16(weights='imagenet')
keras_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
keras_model.summary()

for file in files:
	print file

	feature_path = 'local/feature/' + os.path.splitext(os.path.basename(file))[0]  + '.pkl'

	if(os.path.exists(feature_path)):
		continue

	img = cv2.imread('./local/faces/'+file)
	img = cv2.resize(img, (224, 224))
	img = img[...,::-1]  #BGR 2 RGB
	data = np.array(img, dtype=np.float32)
	data.shape = (1,) + data.shape
	data -= 128
	pred = keras_model.predict(data)[0]
	feature = pred / np.linalg.norm(pred)  # Normalize

	print feature

	pickle.dump(feature, open(feature_path, 'wb'))


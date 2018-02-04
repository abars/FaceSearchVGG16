# ----------------------------------------------
# Face search using VGG16
# ----------------------------------------------

import caffe
from datetime import datetime
import numpy as np
import sys, getopt
import cv2
import os
import glob
import pickle

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model

os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.models import load_model
from keras.preprocessing import image

def interpret_output(output, img_width, img_height):
	classes = ["face"]
	w_img = img_width
	h_img = img_height
	threshold = 0.2
	iou_threshold = 0.5
	num_class = 1
	num_box = 2
	grid_size = 11
	probs = np.zeros((grid_size,grid_size,2,20))
	class_probs = np.reshape(output[0:grid_size*grid_size*num_class],(grid_size,grid_size,num_class))
	scales = np.reshape(output[grid_size*grid_size*num_class:grid_size*grid_size*num_class+grid_size*grid_size*2],(grid_size,grid_size,2))
	boxes = np.reshape(output[grid_size*grid_size*num_class+grid_size*grid_size*2:],(grid_size,grid_size,2,4))
	offset = np.transpose(np.reshape(np.array([np.arange(grid_size)]*grid_size*2),(2,grid_size,grid_size)),(1,2,0))

	boxes[:,:,:,0] += offset
	boxes[:,:,:,1] += np.transpose(offset,(1,0,2))
	boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / grid_size
	boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
	boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])
		
	boxes[:,:,:,0] *= w_img
	boxes[:,:,:,1] *= h_img
	boxes[:,:,:,2] *= w_img
	boxes[:,:,:,3] *= h_img

	for i in range(2):
		for j in range(num_class):
			probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])
	filter_mat_probs = np.array(probs>=threshold,dtype='bool')
	filter_mat_boxes = np.nonzero(filter_mat_probs)
	boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
	probs_filtered = probs[filter_mat_probs]
	classes_num_filtered = np.argmax(probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]

	argsort = np.array(np.argsort(probs_filtered))[::-1]
	boxes_filtered = boxes_filtered[argsort]
	probs_filtered = probs_filtered[argsort]
	classes_num_filtered = classes_num_filtered[argsort]
		
	for i in range(len(boxes_filtered)):
		if probs_filtered[i] == 0 : continue
		for j in range(i+1,len(boxes_filtered)):
			if iou(boxes_filtered[i],boxes_filtered[j]) > iou_threshold : 
				probs_filtered[j] = 0.0
		
	filter_iou = np.array(probs_filtered>0.0,dtype='bool')
	boxes_filtered = boxes_filtered[filter_iou]
	probs_filtered = probs_filtered[filter_iou]
	classes_num_filtered = classes_num_filtered[filter_iou]

	result = []
	for i in range(len(boxes_filtered)):
		result.append([classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

	return result

def iou(box1,box2):
	tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
	lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
	if tb < 0 or lr < 0 : intersection = 0
	else : intersection =  tb*lr
	return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

def show_results(MODE,img,results, img_width, img_height, keras_model, features, img_paths):
	img_cp = img.copy()
	for i in range(len(results)):
		x = int(results[i][1])
		y = int(results[i][2])
		w = int(results[i][3])//2
		h = int(results[i][4])//2

		if(w<h):
			w=h
		else:
			h=w

		xmin = x-w
		xmax = x+w
		ymin = y-h
		ymax = y+h
		if xmin<0:
			xmin = 0
		if ymin<0:
			ymin = 0
		if xmax>img_width:
			xmax = img_width
		if ymax>img_height:
			ymax = img_height

		cv2.rectangle(img_cp,(xmin,ymin),(xmax,ymax),(0,255,0),2)
		cv2.rectangle(img_cp,(xmin,ymin-20),(xmax,ymin),(125,125,125),-1)
		cv2.putText(img_cp,results[i][0] + ' : %.2f' % results[i][5],(xmin+5,ymin-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)	

		target_image=img_cp
		margin=w/4

		x=xmin
		y=ymin
		w*=2
		h*=2

		x2=x-margin
		y2=y-margin
		w2=w+margin*2
		h2=h+margin*2

		if(x2<0):
			x2=0
		if(y2<0):
			y2=0
		if(x2+w2>=img.shape[1]):
			w2=img.shape[1]-1-x2
		if(y2+h2>=img.shape[0]):
			h2=img.shape[0]-1-y2

		face_image = img[y2:y2+h2, x2:x2+w2]

		if(face_image.shape[0]<=0 or face_image.shape[1]<=0):
			continue

		IMAGE_SIZE_KERAS=224

		img_keras = cv2.resize(face_image, (IMAGE_SIZE_KERAS,IMAGE_SIZE_KERAS))
		img_keras = img_keras[...,::-1]  #BGR 2 RGB
		img_keras = np.array(img_keras, dtype=np.float32)
		img_keras.shape = (1,) + img_keras.shape
		img_keras -= 128
		feature = keras_model.predict(img_keras)[0]
		feature = feature / np.linalg.norm(feature)  # Normalize

		cv2.rectangle(target_image, (x2,y2), (x2+w2,y2+h2), color=(0,0,255), thickness=3)
		offset=16

		query = feature
		dists = np.linalg.norm(features - query, axis=1)  # Do search
		ids = np.argsort(dists)[:30] # Top 30 results
		scores = [(dists[id], img_paths[id]) for id in ids]

		for j in range(5):
			prob=scores[j][0]
			file=scores[j][1]

			IMAGE_SIZE_THUMB=128
		
			img = cv2.imread(file)
			img = cv2.resize(img, (IMAGE_SIZE_THUMB,IMAGE_SIZE_THUMB))

			x=j*IMAGE_SIZE_THUMB
			y=i*(IMAGE_SIZE_THUMB+16)
			h, w, _ = img.shape
			target_image[y:y+h, x:x+w] = img

			cv2.putText(target_image, ""+str(prob), (x,y+IMAGE_SIZE_THUMB+16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250));

		if(MODE == "capture"):
			for k in range(1000):
				if(os.path.exists("runtime/faces/"+str(k)+".jpg")):
					continue
				img_keras = cv2.resize(face_image, (IMAGE_SIZE_KERAS,IMAGE_SIZE_KERAS))
				cv2.imwrite("runtime/faces/"+str(k)+".jpg", img_keras)
				feature_path = 'runtime/feature/'+str(k)+'.pkl'
				pickle.dump(feature, open(feature_path, 'wb'))
				features.append(pickle.load(open(feature_path, 'rb')))
				img_paths.append('runtime/' + os.path.splitext(os.path.basename(feature_path))[0] + '.jpg')
				return True

	cv2.imshow('face search',img_cp)
	return False
	
def get_mean(binary_proto,width,height):
	mean_filename=binary_proto
	proto_data = open(mean_filename, "rb").read()
	a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
	mean  = caffe.io.blobproto_to_array(a)[0]
	print "mean value of "+binary_proto+" is "+str(mean)+" shape "+str(mean.shape)

	shape=(mean.shape[0],height,width);
	mean=mean.copy()
	mean.resize(shape)

	print "resized mean value is "+str(mean)

	return mean

def main(argv):
	DEMO_IMG=""
	MODE="search"

	if len(sys.argv) >= 2:
		MODE = sys.argv[1]
		if len(sys.argv) >= 3:
			DEMO_IMG = sys.argv[2]
	else:
		print("usage: python face_search.py [capture/search]")
		sys.exit(1)

	model_filename = './pretrain/face.prototxt'
	weight_filename = './pretrain/face.caffemodel'

	net = caffe.Net(model_filename, weight_filename, caffe.TEST)

	# Read image features
	features = []
	img_paths = []
	for feature_path in glob.glob("local/feature/*"):
	    features.append(pickle.load(open(feature_path, 'rb')))
	    img_paths.append('local/faces/' + os.path.splitext(os.path.basename(feature_path))[0] + '.jpg')
	for feature_path in glob.glob("runtime/feature/*"):
	    features.append(pickle.load(open(feature_path, 'rb')))
	    img_paths.append('runtime/faces/' + os.path.splitext(os.path.basename(feature_path))[0] + '.jpg')

	# Read models
	base_model = VGG16(weights='imagenet')
	keras_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
	keras_model.summary()

	while True:
		cap = cv2.VideoCapture(0)
		ret, frame = cap.read() #BGR
		img=frame
		img = img[...,::-1]  #BGR 2 RGB
		inputs = img.copy() / 255.0
		
		if(DEMO_IMG!=""):
			img = cv2.imread(DEMO_IMG)
			img = img[...,::-1]  #BGR 2 RGB
			inputs = img.copy() / 255.0
		
		transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
		transformer.set_transpose('data', (2,0,1))
		out = net.forward_all(data=np.asarray([transformer.preprocess('data', inputs)]))
		img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		results = interpret_output(out['layer20-fc'][0], img.shape[1], img.shape[0])
		if(show_results(MODE,img_cv,results, img.shape[1], img.shape[0], keras_model, features, img_paths)):
			if(MODE == "capture"):
				break

		k = cv2.waitKey(1)
		if k == 27:
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__=='__main__':
	main(sys.argv[1:])

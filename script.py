#!/usr/bin/env python
import math
import yaml 
from yaml import CLoader as Loader
import numpy as np
import os
import tensorflow as tf
import time
import sys
import random

def loadYaml(path):
	skip_lines = 2
	with open(path) as infile:
		for i in range(skip_lines):
			_ = infile.readline()
		data = yaml.load(infile, Loader=Loader)
		#data = yaml.load(infile)
	
		rangemap = np.array(data['data'])
		rangemap = np.reshape(rangemap,(data['rows'],data['cols'],3))
	return rangemap

def loadDataset(rootDir):
	dataset = []
	labels = []
	count = 0
	start_time = time.time()
	for root,dirs,files in os.walk(rootDir):
		for name in files:
			if(name[-4:] == 'yaml'):
				dataset.append(loadYaml(os.path.join(root,name)))
				labels.append(name[-6])
				if count % 100 == 0:
					t = time.time()-start_time
					print('Loaded %d files in %d minutes and %d seconds'%(count,t/60,t%60))
				count+=1
	return dataset,labels

def removeNan(data, axis):
	indexes = np.where(data == '.Nan')
	
	if axis==2:   #if it's a depth map put nan to inf
		data[indexes] = random.uniform(1.0,1.5)
	else:     #if it's x or y choordinate put nan to real nan
		data[indexes] = np.nan
	
	#convert to float
	data = data.astype(float)
	
	if axis==2:  #inf goes to really large number
		data = np.nan_to_num(data)
	elif axis==0:    #nan interpolated using nearby valid pixels
		mask = np.isnan(data)
		data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
	else:
		data = data.T
		mask = np.isnan(data)
		data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),data[~mask])
		data = data.T
		
	return data
	
	
def convertToFloat(x):
	channel_0 = removeNan(x[:,:,0],0)  #vertical coordinate x
	channel_1 = removeNan(x[:,:,1],1)  #horizontal coordinate y
	channel_2 = removeNan(x[:,:,2],2)   #depth coordinate z
	
	outMatrix = np.empty(x.shape)
	outMatrix[:,:,0] = channel_0
	outMatrix[:,:,1] = channel_1
	outMatrix[:,:,2] = channel_2
	
	return outMatrix

def displayThreeChannels(data):
	fig = plt.figure(figsize=(20,5))

	a=fig.add_subplot(1,3,1)
	a.imshow(data[:,:,0], interpolation = 'nearest')
	b=fig.add_subplot(1,3,2)
	b.imshow(data[:,:,1], interpolation = 'nearest')
	c=fig.add_subplot(1,3,3)
	c.imshow(data[:,:,2], interpolation = 'nearest')

	plt.show()
	
def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value.tolist()))

def save_as_binary(X, Y, filename):
	images = X
	labels = np.array(Y).astype(int)
	num_examples = len(X)
	num_positives = np.count_nonzero(labels)

	print('Saving info in %s'%(filename[:-8]+"txt"))
	info_file = open(filename[:-8]+"txt", 'w+')
	info_file.write("Number of examples %d\n"%num_examples)
	info_file.write("Positives %d\nNegatives %d\n"%(num_positives,num_examples-num_positives))
	info_file.close()

	print("Number of examples %d"%num_examples)
	print("Positives %d, Negatives %d"%(num_positives,num_examples-num_positives))
	
	
	rows = images[0].shape[0]
	cols = images[0].shape[1]
	depth = images[0].shape[2]
	
	print("Image shape (%d,%d,%d)"%(rows,cols,depth))

	writer = tf.python_io.TFRecordWriter(filename)
	shuffled_indexes = list(range(num_examples))
	np.random.shuffle(shuffled_indexes)

	print('Writing %d examples'%len(shuffled_indexes))

	for index in shuffled_indexes:
		image_raw = images[index].tostring()
		example = tf.train.Example(features=tf.train.Features(feature={
			'label': _int64_feature(int(labels[index])),
			'image': _float_feature(images[index].ravel())
		}))
		writer.write(example.SerializeToString())
	writer.close()

def main( srcFolder, outFile):
	X,Y = loadDataset(srcFolder)
	#X,Y = loadDataset('/media/alessio/Data/Projects/KeypointLearningConvolutional/TrainingSet/models/chicken_high/data/0-10/')
	#X,Y = loadDataset('/home/alessio/Scrivania/test/')
	print('Data Loaded\n')
	start_time = time.time()
	for i in range(len(X)):
		X[i] = convertToFloat(X[i])
		if i%100 == 0:
			t = time.time()-start_time
			print('Converted %d image to float in %d minutes and %d seconds'%(i,t/60,t%60))
	print('Data converted\n')

	#splitPoint = math.floor(len(X)*0.80)
	#print('Splitting at %d'%splitPoint)

	#X_train = X[:splitPoint]
	#X_val = X[splitPoint:]

	#Y_train = Y[:splitPoint]
	#Y_val = Y[splitPoint:]

	print('Saving training')
	save_as_binary(X,Y, outFile)
	#save_as_binary(X,Y,'/media/alessio/Data/Projects/KeypointLearningConvolutional/TrainingSet/binary/chicken_high_0_10.tfrecord')
	print('Record Saved\n')
	#print('Saving Validation')
	#save_as_binary(X_val,Y_val,'/media/alessio/Data/Projects/KeypointLearningConvolutional/TrainingSet/binary/cheff_val.tfrecord')
	#save_as_binary(X,Y,'/home/alessio/Scrivania/test/test.tfrecord')
	#print('Record Saved\n')
	print('All Done')

if __name__ == '__main__':
	main(sys.argv[1],sys.argv[2])
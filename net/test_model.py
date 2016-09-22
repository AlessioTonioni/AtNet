#!/usr/bin/env python
import tensorflow as tf
import time
from datetime import datetime
import os
import sys
import numpy as np
import random
from train import *
from loss import *
from model import *
from io_function import *

ROWS = 61
COLS = 61
DEPTH = 1

IMAGE_SHAPE = [ROWS,COLS,3]
WORKING_SHAPE = [ROWS,COLS,DEPTH]

CLASSES = 2

BATCH_SIZE = 1

ONLY_DEPTH = True

def main(root_dir, log_dir):
	#-------------------------------------------------------------------------------
	#                                  SETUP
	#-------------------------------------------------------------------------------

	global_step = tf.Variable(0,trainable=False,name="global_step")
	keep_prob = tf.placeholder(tf.float32, name="drop_prob")

	images_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE,ROWS,COLS,DEPTH],name="image_placeholder")
	labels_ = tf.placeholder(tf.int64, shape=[None], name="labels_plaeholder")

	#read inputs
	filenames_list = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and f[-8:] == 'tfrecord' and "39_42" in f]
	filenames_test = filenames_list
	random.shuffle(filenames_test)
	print('Going to test on ', filenames_test)
	#keep something as validation? 

	#train_batch
	images,labels = read_and_decode_single_example(filenames_test, IMAGE_SHAPE, ONLY_DEPTH)
	#images = augmentData(images)
	images_batch,labels_batch = getShuffledMiniBatch(BATCH_SIZE,images,labels)

	#models
	model = buildDeepNet(images_,keep_prob, BATCH_SIZE,CLASSES,WORKING_SHAPE)

	#create a saver to save and restore models
	saver = tf.train.Saver(tf.all_variables())

	#summary operation for visualization
	summary_op = tf.merge_all_summaries()

	#init operation
	init = tf.initialize_all_variables()

	#accuracy op
	softmaxed = tf.nn.softmax(model)
	correct_prediction = tf.equal(tf.arg_max(softmaxed,1), labels_)
	

	#-----------------------------------------------------------------------------------------------
	#                                          TRAINING
	#-----------------------------------------------------------------------------------------------
	sess = tf.Session()

	sum_writer = tf.train.SummaryWriter(log_dir+'/log/',sess.graph)

	ckpt = tf.train.get_checkpoint_state(log_dir+'/weights/')
	if ckpt and ckpt.model_checkpoint_path:
		print('Founded valid checkpoint file at: '+ckpt.model_checkpoint_path)
		
		#restore variable
		saver.restore(sess, ckpt.model_checkpoint_path)
	
		#restore step
		global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		print('Restored %s step model'%(global_step))
	else:
		print('No checkpoint file found')

	#start the queue runner
	coord = tf.train.Coordinator()
	try:
		threads=[]
		for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
			threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
	
		#num_iter = int(math.ceil(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL/BATCH_SIZE))
		num_iter = 1000
		true_count = 0
		total_sample_Count = num_iter*BATCH_SIZE
		step=0
		predicted_pos = 0
		true_positive = 0
		predicted_neg = 0
		true_negative = 0
		while step<num_iter and not coord.should_stop():
			imgs, lbls = sess.run([images_batch,labels_batch])
			#predictions = sess.run([top_k_op], feed_dict={keep_prob:1.0})
			correct,output = sess.run([correct_prediction,softmaxed], 
				feed_dict={
				keep_prob:1.0,
				images_:imgs,
				labels_:lbls
				})
			#print(res)
			#print(imgs)
			#print(lbls)
			#print(correct)
			#print(output)
			#print('------------------------------')
			if correct:
				true_count +=1
			if np.argmax(output) == 0:
				if correct:
					true_negative +=1
				predicted_neg +=1
			else:
				if correct:
					true_positive +=1
				predicted_pos +=1
			#true_count = np.sum(predictions)
			step+=1
	
		#compute precision
		precision = true_count / step
		#running_acc /= step
		print('%s: precision after %d step @ 1 = %.3f' % (datetime.now(),step, precision))
		print('Predicted Positive: %d - TP %d\nPredicted Negative: %d - TN %d'%(predicted_pos,true_positive,predicted_neg, true_negative))
		#print('%s: precision after %d step @ 1 = %.3f' % (datetime.now(),step, running_acc))
	
		#summary = tf.Summary()
		#summary.ParseFromString(sess.run(summary_op,feed_dict={keep_prob:1.0}))
		#summary.value.add(tag='Precision @ 1', simple_value=acc)
		#summary_writer.add_summary(summary, global_step)
	except Exception as e:
		coord.request_stop(e)

	coord.request_stop()
	coord.join(threads, stop_grace_period_secs=10)
	sess.close()

if __name__=='__main__':
	main(sys.argv[1], sys.argv[2])
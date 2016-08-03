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

ROWS = 112
COLS = 112
DEPTH = 3

IMAGE_SHAPE = [ROWS,COLS,DEPTH]

BATCH_SIZE = 64

CLASSES = 40

LEARNING_RATE = 0.0001

def main(root_dir, logdir):
	#-------------------------------------------------------------------------------
	#                                  SETUP
	#-------------------------------------------------------------------------------
	global_step = tf.Variable(0,trainable=False,name="global_step")
	keep_prob = tf.placeholder(tf.float32, name="drop_prob")
	l_r = tf.placeholder(tf.float32, name="learning_rate")

	images_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE,ROWS,COLS,DEPTH],name="image_placeholder")
	labels_ = tf.placeholder(tf.int64, shape=[None], name="labels_plaeholder")

	#read inputs
	filenames_list = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and f[-9:] == 'tfrecords']
	filenames_train = filenames_list[:-1]
	if(len(filenames_train)>1):
		random.shuffle(filenames_train)
	filenames_val = filenames_list[-1:]
	if(len(filenames_val)>1):
		random.shuffle(filenames_val)
	print('Going to train on ', filenames_train)
	print('Going to test on ', filenames_val)

	#train_batch
	images,labels = read_and_decode_faces(filenames_train,IMAGE_SHAPE)
	images = normalizeData(randomDistortion(randomFlips(images)))
	images_batch,labels_batch = getShuffledMiniBatch(BATCH_SIZE,images,labels)

	#val batch
	images_v,labels_v = read_and_decode_faces(filenames_val, IMAGE_SHAPE)
	images_v = normalizeData(images_v)
	images_batch_v, labels_batch_v = getShuffledMiniBatch(BATCH_SIZE,images_v,labels_v)

	#visualize image in tensorboard
	tf.image_summary('images',images_batch)

	with tf.device("/gpu:1"):
		#models
		model = buildDeepNet(images_,keep_prob, BATCH_SIZE,CLASSES,IMAGE_SHAPE)

		#weights to handle unbalanced training set
		weights = tf.ones([1,CLASSES],tf.float32)

		#loss
		loss = computeLoss(model,labels_,weights,False)

		#train 
		train_op = train(loss,global_step,l_r)

	with tf.device("/cpu:0"):
		#create a saver to save and restore models
		saver = tf.train.Saver(tf.all_variables())

		#accuracy op
		softmaxed = tf.nn.softmax(model)
		correct_prediction = tf.equal(tf.arg_max(softmaxed,1), labels_)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		#placeholder to print average_loss
		average_pl = tf.placeholder(tf.float32)
		average_summary = tf.scalar_summary("average_loss", average_pl)

		#placeholder to print average validation accuracy
		average_val_acc = tf.placeholder(tf.float32)
		val_summary = tf.scalar_summary("average_validation_accuracy",average_val_acc)

		#init operation
		init = tf.initialize_all_variables()

		#summary operation for visualization
		summary_op = tf.merge_all_summaries()


	#-----------------------------------------------------------------------------------------------
	#                                          TRAINING
	#-----------------------------------------------------------------------------------------------
	with tf.Session() as sess:
		with tf.device("/gpu:1"):
			#logging folder 
			if not os.path.exists(logdir+'/log/'):
				os.makedirs(logdir+'/log/')
			if not os.path.exists(logdir+'/weights/'):
				os.makedirs(logdir+'/weights/')
			sum_writer = tf.train.SummaryWriter(logdir+'/log/',sess.graph)
			#init all variables
			sess.run(init)

			#start queue
			tf.train.start_queue_runners(sess=sess)

			losses_history = []
			val_acc_history = []
			for step in range(10000):
				start_time = time.time()
				train_images, train_labels = sess.run([images_batch,labels_batch])

				_, loss_value = sess.run([train_op, loss], 
					feed_dict={
					keep_prob:0.5,
					l_r:LEARNING_RATE,
					images_:train_images,
					labels_:train_labels
					})
				losses_history.append(loss_value)
				duration = time.time() - start_time
			
				assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
			
				if step % 10 == 0:
					num_examples_per_step = BATCH_SIZE
					examples_per_sec = num_examples_per_step / duration
					sec_per_batch = float(duration)
					#print(losses_history,'\n',sum(losses_history),'\n',len(losses_history),'\n')
					format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
					print (format_str % (datetime.now(), step, loss_value,examples_per_sec, sec_per_batch))

					#compute validation score, save averages
					val_images, val_labels = sess.run([images_batch_v,labels_batch_v])
					val_accuracy = sess.run(accuracy, 
						feed_dict={
						keep_prob:1.0, 
						l_r:LEARNING_RATE,
						images_:val_images,
						labels_:val_labels
					})

					val_acc_history.append(val_accuracy)

				if step % 100 == 0:
					#print('Losses_avg: ',sum(losses_history),'\n',len(losses_history),'\n')
					avg_loss = sum(losses_history)/len(losses_history)
					avg_val = sum(val_acc_history)/len(val_acc_history)

					summary_str = sess.run(summary_op, 
						feed_dict={
						keep_prob:1.0, 
						l_r:LEARNING_RATE,
						images_:val_images,
						labels_:val_labels,
						average_pl: avg_loss,
						average_val_acc: avg_val
						})

					train_accuracy = sess.run(accuracy,
						feed_dict={
						keep_prob:1.0, 
						l_r:LEARNING_RATE,
						images_:train_images,
						labels_:train_labels
						})

					print("step %d, avg validation accuracy %g, train_accuracy %g, avg loss %g"%(step,avg_val,train_accuracy, avg_loss))
					sum_writer.add_summary(summary_str, step)
					losses_history = []
					val_acc_history = []

				# Save the model checkpoint periodically.
				if step % 1000 == 0 or (step + 1) == 10000 or (step<1000 and step%100==0):
					checkpoint_path = os.path.join(logdir+'/weights/', 'model.ckpt')
					saver.save(sess, checkpoint_path, global_step=step)

if __name__=='__main__':
	main(sys.argv[1], sys.argv[2])




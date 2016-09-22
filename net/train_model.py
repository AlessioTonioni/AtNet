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

BATCH_SIZE = 64
POS_EXAMPLE_TRAIN = 94179
NEG_EXAMPLE_TRAIN = 75934 
TOTAL_EXAMPLE = POS_EXAMPLE_TRAIN + NEG_EXAMPLE_TRAIN
STEP_PER_EPOCH = TOTAL_EXAMPLE/BATCH_SIZE
LR_COUNTDOWN = 8

CLASSES = 2

LEARNING_RATE = 0.000001

ONLY_DEPTH = True

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
	filenames_list = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and f[-8:] == 'tfrecord']
	filenames_train = [f for f in filenames_list if "39_42" not in f]
	random.shuffle(filenames_train)
	filenames_val = [f for f in filenames_list if "39_42" in f]
	random.shuffle(filenames_val)
	print('Going to train on ', filenames_train)
	print('Going to test on ', filenames_val)

	#train_batch
	images,labels = read_and_decode_single_example(filenames_train, IMAGE_SHAPE, ONLY_DEPTH)
	#images = randomFlips(images)
	images_batch,labels_batch = getShuffledMiniBatch(BATCH_SIZE,images,labels)

	#val batch
	images_v,labels_v = read_and_decode_single_example(filenames_val, IMAGE_SHAPE, ONLY_DEPTH)
	images_batch_v, labels_batch_v = getShuffledMiniBatch(BATCH_SIZE,images_v,labels_v)

	#visualize image in tensorboard
	tf.image_summary('images',images_batch)

	
	#models
	model = buildDeepNet(images_,keep_prob, BATCH_SIZE,CLASSES,WORKING_SHAPE)

	#weights to handle unbalanced training set
	#weights = tf.constant([NEG_EXAMPLE_TRAIN/TOTAL_EXAMPLE,POS_EXAMPLE_TRAIN/TOTAL_EXAMPLE])
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

		#placeholder to print average train accuracy
		average_train_acc = tf.placeholder(tf.float32)
		val_summary = tf.scalar_summary("average_train_accuracy",average_train_acc)

		#init operation
		init = tf.initialize_all_variables()

		#summary operation for visualization
		summary_op = tf.merge_all_summaries()


	#-----------------------------------------------------------------------------------------------
	#                                          TRAINING
	#-----------------------------------------------------------------------------------------------
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		#logging folder 
		if not os.path.exists(logdir+'/log/'):
			os.makedirs(logdir+'/log/')
		if not os.path.exists(logdir+'/weights/'):
			os.makedirs(logdir+'/weights/')
		sum_writer = tf.train.SummaryWriter(logdir+'/log/',sess.graph)

		step = 0
		ckpt = tf.train.get_checkpoint_state(logdir+'/weights/')
		if ckpt and ckpt.model_checkpoint_path:
			print('Founded valid checkpoint file at: '+ckpt.model_checkpoint_path)
			
			#restore variable
			saver.restore(sess, ckpt.model_checkpoint_path)
		
			#restore step
			step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
			print('Restored %d step model'%(step))
		else:
			print('No checkpoint file found, initialization instead')
			#init all variables
			sess.run(init)

		#start queue
		tf.train.start_queue_runners(sess=sess)

		best_loss = np.inf
		lr_countdown = LR_COUNTDOWN
		working_lr = LEARNING_RATE
		losses_history = []
		val_acc_history = []
		train_acc_history = []
		while step < 40000:
			start_time = time.time()
			train_images, train_labels = sess.run([images_batch,labels_batch])

			_, loss_value = sess.run([train_op, loss], 
				feed_dict={
				keep_prob:0.5,
				l_r:working_lr,
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
				format_str = ('%s: epoch %d, step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
				print (format_str % (datetime.now(), step/STEP_PER_EPOCH, step, loss_value,examples_per_sec, sec_per_batch))

				#compute validation score, save average loss
				val_images, val_labels = sess.run([images_batch_v,labels_batch_v])
				val_accuracy = sess.run(accuracy, 
					feed_dict={
					keep_prob:1.0, 
					l_r:working_lr,
					images_:val_images,
					labels_:val_labels
				})

				train_accuracy=sess.run(accuracy,
					feed_dict={
					keep_prob:1.0,
					l_r:working_lr,
					images_:train_images,
					labels_:train_labels
					})

				val_acc_history.append(val_accuracy)
				train_acc_history.append(train_accuracy)

			if step % 100 == 0:
				#print('Losses_avg: ',sum(losses_history),'\n',len(losses_history),'\n')
				avg_loss = sum(losses_history)/len(losses_history)
				avg_val = sum(val_acc_history)/len(val_acc_history)
				avg_train = sum(train_acc_history)/len(train_acc_history)

				if avg_loss<best_loss:
					best_loss = avg_loss
					lr_countdown=LR_COUNTDOWN
				else:
					lr_countdown-=1
					if lr_countdown==0:
						best_loss = avg_loss
						working_lr = working_lr/2
						lr_countdown = LR_COUNTDOWN
					
				summary_str = sess.run(summary_op, 
					feed_dict={
					keep_prob:1.0, 
					l_r:working_lr,
					images_:val_images,
					labels_:val_labels,
					average_pl: avg_loss,
					average_val_acc: avg_val,
					average_train_acc: avg_train
					})

				print("step %d, avg validation accuracy %g, avg train accuracy %g, avg loss %g, learning rate %g, countdown %d"%(step,avg_val,avg_train, avg_loss, working_lr, lr_countdown))
				sum_writer.add_summary(summary_str, step)
				losses_history = []
				val_acc_history = []
				train_acc_history = []

			# Save the model checkpoint periodically.
			if step % 1000 == 0 or (step + 1) == 10000 or (step<1000 and step%100==0):
				checkpoint_path = os.path.join(logdir+'/weights/', 'model.ckpt')
				saver.save(sess, checkpoint_path, global_step=step)

			# Increase step
			step+=1

if __name__=='__main__':
	main(sys.argv[1], sys.argv[2])




import tensorflow as tf

def train(total_loss, global_step, l_r):
	# Variables that affect learning rate.
	#num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
	#decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
	
	# Decay the learning rate exponentially based on the number of steps.
	#lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)
	#tf.scalar_summary('learning_rate', lr)
	
	#train operation
	train_step = tf.train.AdamOptimizer(l_r).minimize(total_loss, global_step=global_step)
	
	return train_step
	

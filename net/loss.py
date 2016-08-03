import tensorflow as tf

def computeLoss(predicted,labels,weights,withAverage=False):
	labels = tf.cast(labels, tf.int64)
	
	#rescale logits by weight of the classe
	weighted_logits = tf.mul(predicted,weights)

	#performs softmax on weighted logits and compute cross entropy 
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		weighted_logits, labels, name='cross_entropy_per_example')
	
	#mean cross entropy for the mini batch
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	with tf.device("/cpu:0"):
		tf.scalar_summary('cross_entropy', cross_entropy_mean)

	#add the cross entropy loss to losses
	tf.add_to_collection('losses',cross_entropy_mean)
	
	#total loss as sum of all the losses
	losses = tf.get_collection('losses')
	loss = tf.add_n(losses, name='total_loss')
	with tf.device("/cpu:0"):
		tf.scalar_summary('loss', loss)
	
	if withAverage:
		#get exponential moving average loss
		loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
		loss_averages_op = loss_averages.apply(losses+[loss])
		tf.scalar_summary('cross_entropy_running_average', loss_averages.average(loss))
		return loss_averages_op
	else:
		return loss

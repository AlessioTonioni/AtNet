import tensorflow as tf
import math

def variable_summaries(var, name):
	with tf.device("/cpu:0"):
		with tf.name_scope("summaries"):
			mean = tf.reduce_mean(var)
			tf.scalar_summary('mean/' + name, mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
		tf.scalar_summary('sttdev/' + name, stddev)
		tf.scalar_summary('max/' + name, tf.reduce_max(var))
		tf.scalar_summary('min/' + name, tf.reduce_min(var))
		tf.histogram_summary(name, var)

def visualize_weights(weights,name):
	with tf.device("/cpu:0"):
		x_min = tf.reduce_min(weights)
		x_max = tf.reduce_max(weights)
		weights_0_to_1 = (weights - x_min) / (x_max - x_min)
		weights_0_to_255_uint8 = tf.image.convert_image_dtype (weights_0_to_1, dtype=tf.uint8)

		# to tf.image_summary format [batch_size, height, width, channels]
		weights_transposed = tf.transpose (weights_0_to_255_uint8, [3, 0, 1, 2])

		# this will display random 6 filters 
		tf.image_summary(name, weights_transposed, max_images=6)

#weights + optional L2 regolarization
def variable_with_weight_decay(n, s, wd=0.0):
	var = tf.get_variable(n, shape=s, initializer=tf.contrib.layers.xavier_initializer())
	if wd != 0.0:
		weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses',weight_decay)
	return var

def convRelu(name, previous_layer, kernelShape, stride, visualize=False, withLeaky = False, alphaLeaky = 1/100):
	with tf.variable_scope(name) as scope:
		kernel = variable_with_weight_decay(name+"/weights",kernelShape,wd=0.0)
		if(visualize):
			visualize_weights(kernel,name+"/weights")
		variable_summaries(kernel,name+"/weights")
		biases = tf.Variable(tf.constant(0.1, shape=[kernelShape[3]]),name=name+"/bias")
		
		#conv +bias
		conv = tf.nn.conv2d(previous_layer,kernel,stride,padding='SAME')
		bias = tf.nn.bias_add(conv,biases)
		
		#relu
		if withLeaky:
			rectified = tf.maximum(alphaLeaky*bias,bias)
		else:	
			rectified = tf.nn.relu(bias)
		
		return rectified

def pool(name,previous_layer,poolShape,stride=[1, 2, 2, 1]):
	with tf.variable_scope(name) as scope:
		
		#max-pooling
		pool = tf.nn.max_pool(previous_layer,poolShape,strides=stride, padding='SAME')
		
		return pool

def fullyConnectedConv(name,previous_layer,kernelShape):
	with tf.variable_scope(name) as scope:
		kernel = variable_with_weight_decay(name+"/weights",kernelShape, wd=0.0)
		variable_summaries(kernel, name+"/weights")
		biases = tf.Variable(tf.constant(0.1,shape=[kernelShape[3]]),name=name+"/bias")
		
		#conv+bias
		conv = tf.nn.conv2d(previous_layer,kernel,[1,1,1,1],padding='VALID')
		bias = tf.nn.bias_add(conv,biases)
		
		#relu
		rectified = tf.nn.relu(bias)
		
		return rectified

def softMaxConv(name, previous_layer, kernelShape):
	with tf.variable_scope(name) as scope:
		kernel = variable_with_weight_decay(name+"/weights",kernelShape, wd=0.0)
		variable_summaries(kernel,name+"/weights")
		biases = tf.Variable(tf.constant(0.1,shape=[kernelShape[3]]), name=name+"/bias")
		
		#conv+bias
		conv = tf.nn.conv2d(previous_layer,kernel,[1,1,1,1],padding='VALID')
		bias = tf.nn.bias_add(conv,biases)
		
		#softmax
		#y = tf.nn.softmax(bias)
		
		return bias
		#return y

#------------------------------------------------------------------#
#                    Complete network models                       #
#------------------------------------------------------------------#
def buildNet(images, keepProb, batch_size, num_output, image_size):
	
	#conv1 61x61x3==>61x61x32
	conv1 = convRelu("conv1", images, [5,5,image_size[2],32],[1,1,1,1],True)

	#pool1 61x61x32==>31x31x32
	pool1 = pool("pool1",conv1,[1,2,2,1],[1,2,2,1])
	
	#conv2 31x31x32 ==> 31x31x64
	conv2 = convRelu("conv2",pool1,[5,5,32,64],[1,1,1,1])
	
	#conv3 31x31x64 ==> 31x31x128
	conv3 = convRelu("conv3",conv2,[3,3,64,128],[1,1,1,1])
	
	#pool2 31x31x128 ==> 16x16x128
	pool2 = pool("pool2", conv3, [1,2,2,1],[1,2,2,1])
	
	kernel_side = math.ceil(image_size[0]/4)

	#fc1
	fc1 = fullyConnectedConv("fc1",pool2, [kernel_side,kernel_side,128,512])
	
	#dropout
	fc1_drop = tf.nn.dropout(fc1, keepProb)
	
	#fc2
	fc2 = fullyConnectedConv("fc2",fc1_drop, [1,1,512,1024])
	
	#softmax
	sm = softMaxConv("soft_max",fc2,[1,1,1024,num_output])
	
	#reshape for loss
	reshape = tf.reshape(sm, [batch_size, -1])
	
	return reshape

def buildDeepNet(images, keepProb, batch_size, num_output, image_size):
	
	#conv1 61x61x3==>61x61x32
	conv1 = convRelu("conv1", images, [3,3,image_size[2],32],[1,1,1,1],True)

	#pool1 61x61x32==>31x31x32
	pool1 = pool("pool1",conv1,[1,2,2,1],[1,2,2,1])
	
	#conv2 31x31x32 ==> 31x31x64
	conv2 = convRelu("conv2",pool1,[3,3,32,64],[1,1,1,1])
	
	#conv3 31x31x64 ==> 31x31x128
	conv3 = convRelu("conv3",conv2,[3,3,64,128],[1,1,1,1])
	
	#pool2 31x31x128 ==> 16x16x128
	pool2 = pool("pool2", conv3, [1,2,2,1],[1,2,2,1])

	conv4 = convRelu("conv4",pool2,[3,3,128,256],[1,1,1,1])

	conv5 = convRelu("conv5",conv4,[3,3,256,512],[1,1,1,1])

	pool3 = pool("pool3",conv5,[1,2,2,1],[1,2,2,1])
	
	kernel_side = math.ceil(image_size[0]/8)

	#fc1
	fc1 = fullyConnectedConv("fc1",pool3, [kernel_side,kernel_side,512,512])
	
	#dropout
	fc1_drop = tf.nn.dropout(fc1, keepProb)
	
	#fc2
	fc2 = fullyConnectedConv("fc2",fc1_drop, [1,1,512,1024])
	
	#softmax
	sm = softMaxConv("soft_max",fc2,[1,1,1024,num_output])
	
	#reshape for loss
	reshape = tf.reshape(sm, [batch_size, -1])
	
	return reshape

def buildLeNet(images, keepProb, batch_size, num_output, image_size):

	#conv1 61x61x1==>61x61x20
	conv1 = convRelu("conv1", images, [5,5,image_size[2],20],[1,1,1,1],True)

	#pool1 61x61x20==>31x31x20
	pool1 = pool("pool1",conv1,[1,2,2,1],[1,2,2,1])

	#conv2 31x31x20==>31x31x50
	conv2 = convRelu("conv2", pool1, [5,5,20,50],[1,1,1,1],False)

	#pool2 31x31x50=>16x16x50
	pool2 = pool("pool2", conv2, [1,2,2,1],[1,2,2,1])

	kernel_side = math.ceil(image_size[0]/4)

	#fc1
	fc1 = fullyConnectedConv("fc1",pool2, [kernel_side,kernel_side,50,500])

	#dropout
	fc1_drop = tf.nn.dropout(fc1, keepProb)
	
	#softmax
	sm = softMaxConv("soft_max",fc1_drop,[1,1,500,num_output])

	#reshape for loss
	reshape = tf.reshape(sm, [batch_size, -1])
	
	return reshape
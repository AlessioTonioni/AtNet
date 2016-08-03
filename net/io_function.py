import tensorflow as tf

def read_and_decode_faces(filenames, image_shape):
	# first construct a queue containing a list of filenames.
	# this lets a user split up there dataset in multiple files to keep
	# size down
	filename_queue = tf.train.string_input_producer(filenames,num_epochs=None)

	#symbolic reader to read one example at a time
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
		serialized_example,
		# Defaults are not specified since both keys are required.
		features={
			'image_raw': tf.FixedLenFeature([], tf.string),
			'label': tf.FixedLenFeature([], tf.int64),
		}
	)

	image = tf.image.decode_jpeg(features['image_raw'], channels=3)
	image.set_shape(image_shape)
	label = features['label']

	return image,label

def read_and_decode_single_example(filenames, image_shape, onlyDepth = False):
	# first construct a queue containing a list of filenames.
	# this lets a user split up there dataset in multiple files to keep
	# size down
	filename_queue = tf.train.string_input_producer(filenames,num_epochs=None)
	
	#symbolic reader to read one example at a time
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
		serialized_example,
		# Defaults are not specified since both keys are required.
		features={
			'image': tf.FixedLenFeature([image_shape[0]*image_shape[1]*image_shape[2]], tf.float32),
			'label': tf.FixedLenFeature([], tf.int64),
		}
	)

	# Convert from a scalar list to a proper shaped tensor
	image = features['image']
	image = tf.reshape(image,image_shape)
	if onlyDepth:
		channels = tf.split(2,3,image)
		image = channels[2]
	label = features['label']
	
	return image,label

def getShuffledMiniBatch(num_examples,image,label):
	images_batch, labels_batch = tf.train.shuffle_batch(
		[image,label],
		batch_size=num_examples,
		capacity=2000,
		min_after_dequeue=1000)
	
	return images_batch, labels_batch


def randomFlips(image):
	#geometric distortion
	distorted_image = tf.image.random_flip_left_right(image)
	#distorted_image = tf.image.random_flip_up_down(distorted_image)
	
	return distorted_image

def normalizeData(image):
	#normalize the image subtracting the mean
	normalized_image = tf.image.per_image_whitening(image)

	return normalized_image

def randomDistortion(image):
	#perform random brightness and contrast perturbation
	distorted_image = tf.image.random_brightness(image,max_delta=63)
	distorted_image = tf.image.random_contrast(image, lower=0.2,upper=1.8)

	return distorted_image
##
## A Modern implementatio of Yann Lecun's LeNet-5 Convolutional Network
## The overall architecture here is the same, but we've made some changes 
## to make LeNet-5 follow modern practices a little more closely:
##
## 1. Activation functions immediately follow conv layers, not pooling layers. 
## 2. Our pooling layers don't have learnable parameters, unlike LeCun's sub-sampling layers 
## 3. We're using a cross entropy cost function, LeNet used radial basis functions 
## 4. We're using the Adam optimizer, LeCun used gradient descent with a variable learning rate.
##

import tensorflow as tf


class Model(object):
	def __init__(self, X, y):
		'''
		Setup graph using class. 
		X = input tensor
		y = output tensor
		'''

		## Some hyper paramters
		self.learning_rate = 1e-3
		self.minibatch_size = 128
		self.num_epochs = 100
		self.evaluation_frequency = 1000

		# We're going to be using tensorflow's tf.layers.conv2d to implement our convolutions
		# This method expects a 4 dimensional tensor [batch size, im height, im width, color channels]
		# In our our training script we've already "unrolled" our images, 
		# So we'll have to reshape them here. It's a little inefficient to unroll and 
		# then reshape our images, but it allows us to keep our overall code a little 
		# more consistent between mnist examples. 

		X_reshaped = tf.reshape(X, [-1, 28, 28, 1])

		C1 = tf.layers.conv2d(inputs = X_reshaped, 
							  filters = 6, 
							  kernel_size = (5, 5), 
							  activation = 'tanh',
							  name = 'C1', 
							  padding = 'same')

		#The Original LeNet learns an averaging pooling layer, 
		#here we'll use max pooling instead.
		S2 = tf.layers.average_pooling2d(inputs = C1, 
									 pool_size = (2,2), 
									 strides = (2, 2), 
									 name = 'S2')

		C3 = tf.layers.conv2d(inputs = S2, 
			                  filters = 16, 
			                  kernel_size = (5,5), 
			                  activation = 'tanh',
			                  name = 'C3')

		S4 = tf.layers.average_pooling2d(inputs = C3, 
			                         pool_size = (2,2), 
			                         strides = (2, 2), 
			                         name = 'S4')

		S4_unrolled = tf.reshape(S4, [-1, 400])

		C5 = tf.layers.dense(inputs = S4_unrolled, 
							 units = 120, 
							 activation = 'tanh',
							 name = 'C5') 

		F6 = tf.layers.dense(inputs = C5, 
							 units = 84, 
							 activation = 'tanh', 
							 name = 'F6')


		logits = tf.layers.dense(inputs = F6, 
								 units = 10, 
								 activation = None, 
								 name = 'output')

		self.yhat = tf.nn.softmax(logits)

		# Cross Entropy Cost Function
		self.cost = tf.losses.softmax_cross_entropy(y, logits)

		# As we train, it will also be nice to keep track of the accuracy of our classifier
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.yhat, 1)) # Check if predictions are equal to labels
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # Compute average accuracy

		# Add Optimizer to Graph:
		optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
		self.train_op = optimizer.minimize(self.cost)

		#Setup Summary Writing for Tensorboard:
		tf.summary.scalar(name = 'cost', tensor = self.cost)
		tf.summary.scalar(name = 'accuracy', tensor = self.accuracy)
		self.merged_summary_op = tf.summary.merge_all() #Merges all summaries, in this case we only have one!




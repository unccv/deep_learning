##
## Simple single layer model with mean square error cost function. 
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
		self.learning_rate = 1e-2
		self.minibatch_size = 128
		self.num_epochs = 200
		self.evaluation_frequency = 1000

		# Single fully connected layer with a sigmoid activation function:
		self.yhat = tf.layers.dense(inputs = X, units = 10, activation = 'sigmoid', name = 'fc_1')

		# Mean Square Error
		self.cost = tf.reduce_mean(tf.squared_difference(y, self.yhat))

		# As we train, it will also be nice to keep track of the accuracy of our classifier
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.yhat, 1)) # Check if predictions are equal to labels
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # Compute average accuracy

		# Add Optimizer to Graph:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)
		self.train_op = optimizer.minimize(self.cost)

		#Setup Summary Writing for Tensorboard:
		tf.summary.scalar(name = 'cost', tensor = self.cost)
		tf.summary.scalar(name = 'accuracy', tensor = self.accuracy)
		self.merged_summary_op = tf.summary.merge_all() #Merges all summaries, in this case we only have one!




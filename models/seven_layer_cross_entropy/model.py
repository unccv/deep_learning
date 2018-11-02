##
## Seven layer fully connected CNN
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

		fc1 = tf.layers.dense(inputs = X, units = 128, activation = 'sigmoid', name = 'fc_1')
		fc2 = tf.layers.dense(inputs = fc1, units = 64, activation = 'sigmoid', name = 'fc_2')
		fc3 = tf.layers.dense(inputs = fc2, units = 64, activation = 'sigmoid', name = 'fc_3')
		fc4 = tf.layers.dense(inputs = fc3, units = 32, activation = 'sigmoid', name = 'fc_4')
		fc5 = tf.layers.dense(inputs = fc4, units = 32, activation = 'sigmoid', name = 'fc_5')
		fc6 = tf.layers.dense(inputs = fc5, units = 16, activation = 'sigmoid', name = 'fc_6')
		logits = tf.layers.dense(inputs = fc6, units = 10, activation = None, name = 'fc_7')
		self.yhat = tf.nn.softmax(logits)

		# Cost Function
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




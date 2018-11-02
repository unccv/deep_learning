##
## General mnist Neural Network Training Script
##

import numpy as np
import tensorflow as tf
import sys, argparse

from generator import Generator


def train(model_dir):
	'''
	Train mnist classifier.
	model_file = name of .py model file to import and train. 
	'''

	#Load mnist dataset
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

	#Normalize Input Data between 0 and 1
	x_train = x_train.astype('float32')/255
	x_test = x_test.astype('float32')/255

	#Unroll examples into one row for each example
	x_train = np.reshape(x_train, (-1, 784))
	x_test = np.reshape(x_test, (-1, 784))

	#use built in one-hot encoding converter:
	y_train = tf.keras.utils.to_categorical(y_train, 10)
	y_test = tf.keras.utils.to_categorical(y_test, 10)

	# We'll pass in our images as unrolled vectors
	# Passing in None for our first dimension allows our graph to accept a variable size here.
	# The name argument is optional, but will make things more clear when we visualize our graph with tensorboard.
	X = tf.placeholder(dtype = tf.float32, shape = (None, 28*28), name = 'X') 
	y = tf.placeholder(dtype = tf.float32, shape = [None, 10], name = 'y') #One output dimension for each class

	#Import model class 
	sys.path.append('models/' + model_dir)
	from model import Model

	#Setup Model
	M = Model(X, y)

	#Setup data generators for stochastic gradient descent:
	G = Generator(X = x_train, y = y_train, minibatch_size = M.minibatch_size) # Generator for training data
	GT = Generator(X = x_test, y = y_test, minibatch_size = M.minibatch_size) # Generator for testing data

	#Setup a new session
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	#Write tensorflow log file to tf_data directory:
	train_writer = tf.summary.FileWriter(logdir = 'models/' + model_dir + '/train', graph = sess.graph)
	test_writer = tf.summary.FileWriter(logdir = 'models/' + model_dir + '/test', graph = sess.graph)

	print('Training...')
	global_step = 0
	while G.num_epochs < M.num_epochs:
		#Take Gradient Descent Step:
	    G.generate()
	    sess.run(M.train_op, feed_dict = {X: G.X, y: G.y})
	    
	    if global_step % M.evaluation_frequency == 0:
	        summary = sess.run(M.merged_summary_op, feed_dict = {X: G.X, y: G.y})
	        train_writer.add_summary(summary, global_step)
	        train_writer.flush() #Go ahead and write to tfevent file so we can visualize as we train
	        
	        GT.generate()
	        summary = sess.run(M.merged_summary_op, feed_dict = {X: GT.X, y: GT.y})
	        test_writer.add_summary(summary, global_step)
	        test_writer.flush()

	        test_set_accuracy = sess.run(M.accuracy, feed_dict = {X: x_test, y: y_test})
	        
	        print(str(global_step), 'steps, ',  
	        	  str(G.num_epochs), ' epochs, test set accuracy = ', 
	        	  test_set_accuracy)

	    global_step += 1




### --------------------------------- ###
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='(python train.py -model_dir PATH_TO_MODEL_DIRECTORY')
	parser.add_argument("-d", "-model_dir", dest='model_dir', required=True, help='Path to training directory')
	args = parser.parse_args()
	train(args.model_dir)
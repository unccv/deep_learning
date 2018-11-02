"""
Alenet implmentation from Frederik Kratzert modified by Stephen Welch.

This is an TensorFLow implementation of AlexNet by Alex Krizhevsky at all.

Paper:
(http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

Explanation can be found in blog post:
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

This script enables finetuning AlexNet on any given Dataset with any number of
classes. The structure of this script is strongly inspired by the fast.ai
Deep Learning class by Jeremy Howard and Rachel Thomas, especially their vgg16
finetuning script:
Link:
- https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16.py


The pretrained weights can be downloaded here and should be placed in the same
folder as this file:
- http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

@author: Frederik Kratzert (contact: f.kratzert(at)gmail.com)
"""

import tensorflow as tf
import numpy as np


class AlexNet(object):
    """Implementation of the AlexNet."""

    def __init__(self, X, keep_prob, num_layers_to_load, weights_path='DEFAULT'):
        """Create the graph of the AlexNet model.

        Args:
            X: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_layers_to_load: how many of alexnet's 8 layers 
            build the graph for and laod weights for
            weights_path: Complete path to the pretrained weight file, if it
                isn't in the same folder as this code
        """
        # Parse input arguments into class variables
        self.X = X
        self.KEEP_PROB = keep_prob
        self.NUM_LAYERS_TO_LOAD = num_layers_to_load

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        """Create the network graph."""
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        self.conv1 = conv(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        self.norm1 = lrn(self.conv1, 2, 2e-05, 0.75, name='norm1')
        self.pool1 = max_pool(self.norm1, 3, 3, 2, 2, padding='VALID', name='pool1')
        
        self.all_variable_names = ['conv1', 'norm1', 'pool1']

        if self.NUM_LAYERS_TO_LOAD > 1:
            # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
            self.conv2 = conv(self.pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
            self.norm2 = lrn(self.conv2, 2, 2e-05, 0.75, name='norm2')
            self.pool2 = max_pool(self.norm2, 3, 3, 2, 2, padding='VALID', name='pool2')
            self.all_variable_names.extend(['conv2', 'norm2', 'pool2'])
        
        if self.NUM_LAYERS_TO_LOAD > 2:
            # 3rd Layer: Conv (w ReLu)
            self.conv3 = conv(self.pool2, 3, 3, 384, 1, 1, name='conv3')
            self.all_variable_names.extend(['conv3'])

        if self.NUM_LAYERS_TO_LOAD > 3:
            # 4th Layer: Conv (w ReLu) splitted into two groups
            self.conv4 = conv(self.conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')
            self.all_variable_names.extend(['conv4'])

        if self.NUM_LAYERS_TO_LOAD > 4:
            # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
            self.conv5 = conv(self.conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
            self.pool5 = max_pool(self.conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
            self.all_variable_names.extend(['conv5', 'pool5'])

        if self.NUM_LAYERS_TO_LOAD > 5:
            # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
            flattened = tf.reshape(self.pool5, [-1, 6*6*256])
            self.fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
            self.dropout6 = dropout(self.fc6, self.KEEP_PROB)
            self.all_variable_names.extend(['fc6'])

        if self.NUM_LAYERS_TO_LOAD > 6:
            # 7th Layer: FC (w ReLu) -> Dropout
            self.fc7 = fc(self.dropout6, 4096, 4096, name='fc7')
            self.dropout7 = dropout(self.fc7, self.KEEP_PROB)
            self.all_variable_names.extend(['fc7'])

        if self.NUM_LAYERS_TO_LOAD > 7:
            # 8th Layer: FC and return unscaled activations
            self.fc8 = fc(self.dropout7, 4096, 1000, relu=False, name='fc8')
            self.all_variable_names.extend(['fc8'])

    def load_initial_weights(self, session):
        """Load weights from file into network.

        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if op_name in self.all_variable_names:
                print('Loading ', op_name, ' values...')

                with tf.variable_scope(op_name, reuse=True):

                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))

                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))

                        


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)

##
## Evaluation Script
##

import numpy as np

from sample_model import Model

from data_loader import data_loader
from generator import Generator


#Load up some data for evaluation:
label_indices = {'brick': 0, 'ball': 1, 'cylinder': 2}
channel_means = np.array([147.12697, 160.21092, 167.70029])
data_path = '../data/hard_large'
minibatch_size = 32
num_batches_to_test = 10
checkpoint_dir = 'tf_data/sample_model'

data = data_loader(label_indices = label_indices, 
           		   channel_means = channel_means,
           		   train_test_split = 0.5, 
           		   data_path = data_path)

#Instantiate Model
M = Model(mode = 'test')

#Evaluate on test images:
GT = Generator(data.test.X, data.test.y, minibatch_size = minibatch_size)

num_correct = 0
num_total = 0

for i in range(num_batches_to_test):
	GT.generate()
	yhat = M.predict(X = GT.X, checkpoint_dir = checkpoint_dir)
	correct_predictions = (np.argmax(yhat, axis = 1) == np.argmax(GT.y, axis = 1))
	num_correct += np.sum(correct_predictions)
	num_total += len(correct_predictions)

print('Accuracy = ', num_correct/num_total)




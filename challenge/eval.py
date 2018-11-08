##
## Evaluation Script
##

import numpy as np
import time
from sample_model import Model
from data_loader import data_loader
from generator import Generator


def evaluate(label_indices = {'brick': 0, 'ball': 1, 'cylinder': 2},
             channel_means = np.array([147.12697, 160.21092, 167.70029]),
             data_path = '../data',
             minibatch_size = 32,
             num_batches_to_test = 10,
             checkpoint_dir = 'tf_data/sample_model'):

    
    print("1. Loading data")
    data = data_loader(label_indices = label_indices, 
               		   channel_means = channel_means,
               		   train_test_split = 0.5, 
               		   data_path = data_path)

    print("2. Instantiating the model")
    M = Model(mode = 'test')

    #Evaluate on test images:
    GT = Generator(data.test.X, data.test.y, minibatch_size = minibatch_size)
    
    num_correct = 0
    num_total = 0
    
    print("3. Evaluating on test images")
    for i in range(num_batches_to_test):
        GT.generate()
        yhat = M.predict(X = GT.X, checkpoint_dir = checkpoint_dir)
        correct_predictions = (np.argmax(yhat, axis = 1) == np.argmax(GT.y, axis = 1))
        num_correct += np.sum(correct_predictions)
        num_total += len(correct_predictions)
    
    accuracy =  round(num_correct/num_total,4)

    return accuracy
   
def calculate_score(accuracy):
    score = 0
    if accuracy >= 0.92:
       score = 10
    elif accuracy >= 0.9:
       score = 9
    elif accuracy >= 0.85:
       score = 8
    elif accuracy >= 0.8:
       score = 7
    elif accuracy >= 0.75:
       score = 6
    elif accuracy >= 0.70:
       score = 5
    else:
       score = 4
    return score

if __name__ == '__main__':
    program_start = time.time()
    accuracy = evaluate()
    score = calculate_score(accuracy)
    program_end = time.time()
    total_time = round(program_end - program_start,2)
    print()
    print("Execution time (seconds) = ", total_time)
    print('Accuracy = ' + str(accuracy))
    print("Score = ", score)
    print()


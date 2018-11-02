##
## generator.py
## Generator random minibatches for stochastic gradient descent. 
## This method only works if all your data fits in RAM. 
##



import numpy as np

class Generator(object):
    '''
    Minibatch generator class. 
    '''
    def __init__(self, X, y, minibatch_size):
        '''
        X = array of all training or testing data
        y = array of all training or testing labels
        '''
        self.all_X = X
        self.all_y = y
        self.minibatch_size = minibatch_size
        
        self.indices = np.arange(self.all_X.shape[0])
        np.random.shuffle(self.indices)
        self.pointer = 0   
        self.num_epochs = 0
        
    def generate(self):
        '''
        Make that minibatch!
        '''
        self.X = self.all_X[self.indices[self.pointer:self.pointer+self.minibatch_size]]
        self.y = self.all_y[self.indices[self.pointer:self.pointer+self.minibatch_size]]
        self.pointer += self.minibatch_size
        #End of Epoch
        if self.pointer > self.all_X.shape[0] - self.minibatch_size:
            self.pointer = 0
            self.num_epochs += 1
            np.random.shuffle(self.indices) #Reshuffle indices at the end of each epoch
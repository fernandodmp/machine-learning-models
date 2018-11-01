import numpy as np

ts_inputs = np.array([[0,0,1,0],
                      [1,1,1,0],
                      [1,0,1,1],
                      [0,1,1,1],
                      [0,1,0,1],
                      [1,1,1,1],
                      [0,0,0,0]])

ts_outputs = np.array([[0,1,1,0,0,1,0]]).T

class Perceptron():
    def __init__(self):
        self.synapse_weights = np.random.rand(4,1)
    
    def sigmoid_function(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid_derivative(x) * (self.sigmoid_function(x) - 1) 
    
    def train(self, inputs, expected_outputs, n_iterations = 10, learning_rate = 0):
        pass
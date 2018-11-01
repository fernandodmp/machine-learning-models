import numpy as np

class Perceptron():
    def __init__(self):
        self.synapse_weights = np.random.rand(4,1)
    
    def sigmoid_function(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid_function(x) * (1 - self.sigmoid_function(x)) 
    
    def fit(self, inputs, expected_outputs, n_iterations = 10, learning_rate = 0.5):
        delta_weights = np.zeros((len(inputs[0]),len(inputs)))
        for iteration in range(n_iterations):
            print("Iteration #{}".format(iteration + 1))
            x = np.dot(inputs, self.synapse_weights)
            activation = self.sigmoid_function(x)
            for i in range(len(inputs)):
                #cost = (activation[i] - expected_outputs[i]) ** 2
                cost_prime = 2 * (activation[i] - expected_outputs[i])
                for j in range(4):
                    delta_weights[j][i] = cost_prime * inputs[i][j] * self.sigmoid_derivative(x[i])
                    
    
        delta_avg = np.array([np.average(delta_weights, axis = 1)]).T
        self.synapse_weights = self.synapse_weights - delta_avg * learning_rate

    def predict(self, inputs):
        return self.sigmoid_function(np.dot(inputs, self.synapse_weights))




from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate, epochs, use_bias, activation_function):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.use_bias = use_bias
        self.activation_function = activation_function

        self.weights = self.initialize_weights()
        self.biases = self.initialize_biases()
        self.activations = None
        

    def initialize_weights(self):
        weights = []
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        print(layer_sizes)
        for i in range(1, len(layer_sizes)):
            weight_matrix = np.random.rand(layer_sizes[i], layer_sizes[i - 1])
            weights.append(weight_matrix)
        return weights

    def initialize_biases(self):
        if self.use_bias:
            biases = [np.zeros((layer_size, 1)) for layer_size in self.hidden_layers + [self.output_size]]
            return biases
        else:
            return [None] * len(self.weights)

    def tanh(x):
        # tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
        exp_positive = math.exp(2 * x)
        exp_negative = 1 / exp_positive
        return (exp_positive - 1) / (exp_positive + 1)

    def activate(self, z):
        if self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif self.activation_function == "tanh":
            return tanh(z)

    def activate_derivative(self, a):
        if self.activation_function == "sigmoid":
            return a * (1 - a)
        elif self.activation_function == "tanh":
            return 1 - np.power(a, 2)

    def forward_propagation(self, X):
        activations = [X]
        dot_products = []

        for i in range(len(self.weights)):
            dot_product = np.dot(self.weights[i], activations[-1])  # try np.dot(self.weights[i],  X   )
            # the dot_product is a vector resulted from multiplying a matrix by a vector
            if self.use_bias:
                dot_product += self.biases[i]
            dot_products.append(dot_product)
            activation = self.activate(dot_product)
            activations.append(activation)

        self.activations = activations
        return activations, dot_products

    def backward_propagation(self, X, y, activations, dot_products):
        gradients = [None] * len(self.weights)
        delta = y - activations[-1]

        # print(y, '\n', activations[-1], '\n', delta, '\n*******************\n')
        gradients[-1] = delta * self.activate_derivative(activations[-1])
        # Now U computed the errors signal of the last layer (output layer)
        # and saved it in the last ind. of the gradients array
        for i in range(len(self.weights) - 2, -1, -1):  # starts from the matrix b4 the last one and iterates
            # backward till the first matrix weights[0]
            # print("", i,  self.weights[i+1], gradients[i + 1], self.activate_derivative(activations[i+1]), sep="\n\n")

            gradients[i] = np.dot(self.weights[i + 1].T, gradients[i + 1]) * self.activate_derivative(activations[i + 1])

        # print(gradients)
        return gradients

    def update_weights(self, gradients, activations):
        for i in range(len(self.weights)):  # the weight Matrices
            for u in range(len(self.weights[i])):   # the Rows of each matrix
                for v in range(len(self.weights[i][u])):    # the Columns of each Row
                    self.weights[i][u][v] += self.learning_rate * gradients[i][u] * activations[i][v]

    def train(self, X_train, y_train):
        for _ in range(self.epochs):
            for i in range(len(X_train)):
                X = X_train[i]
                y = y_train[i]

                activations, weighted_inputs = self.forward_propagation(X)
                if(i == len(X_train)-1 or i == 0):
                    print(activations, '\n---------\n')
                gradients = self.backward_propagation(X, y, activations, weighted_inputs)
                self.update_weights(gradients, activations)
            # print(self.weights[-1], '\n-----------------')
            

    # def predict(self, X_test):
    #     predictions = []
    #     for i in range(len(X_test)):
    #         X = X_test[i]
    #         activations, _ = self.forward_propagation(X)
    #         print(activations[-1])
    #         predictions.append(activations[-1])
    #
    #     return (predictions)

    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            X = X_test[i]
            activations, _ = self.forward_propagation(X)
            max_index = np.argmax(activations[-1])
            result_array = np.zeros_like(activations[-1])
            result_array[max_index] = 1
            # print(result_array)

            predictions.append(result_array)

        return (predictions)

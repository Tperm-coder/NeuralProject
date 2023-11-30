from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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
        for i in range(1, len(layer_sizes)):
            weight_matrix = np.random.rand(layer_sizes[i], layer_sizes[i - 1]) * 0.01
            weights.append(weight_matrix)
        return weights

    def initialize_biases(self):
        if self.use_bias:
            biases = [np.zeros((layer_size, 1)) for layer_size in self.hidden_layers + [self.output_size]]
            return biases
        else:
            return [None] * len(self.weights)

    def activate(self, z):
        if self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif self.activation_function == "tanh":
            return np.tanh(z)

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
        m = X.shape[0]
        gradients = [None] * len(self.weights)
        delta = y - activations[-1]
        gradients[-1] = delta * self.activate_derivative(activations[-1])
        # Now U computed the errors signal of the last layer (output layer)
        # and saved it in the last ind. of the gradients array
        for i in range(len(self.weights) - 2, -1, -1):  # starts from the matrix b4 the last one and iterates
            # backward till the first matrix weights[0]
            # print("", i,  self.weights[i+1], gradients[i + 1], self.activate_derivative(activations[i+1]), sep="\n\n")
            gradients[i] = np.dot(self.weights[i + 1].T, gradients[i + 1]) * self.activate_derivative(
                activations[i + 1])

        # print(gradients)
        return gradients

    def update_weights(self, gradients):
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * gradients[i][:, np.newaxis] * self.activations[i]

    def train(self, X_train, y_train):
        for epoch in range(self.epochs):
            for i in range(len(X_train)):
                X = X_train[i]
                y = y_train[i]

                activations, weighted_inputs = self.forward_propagation(X)
                gradients = self.backward_propagation(X, y, activations, weighted_inputs)
                self.update_weights(gradients)

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
            print(result_array)

            predictions.append(result_array)

        return (predictions)


# data preprocessing & splitting
data = pd.read_csv('data.csv')
df = pd.DataFrame(data)

MinorAxisLength = df['MinorAxisLength']
ave = MinorAxisLength.mean()
MinorAxisLength.fillna(ave, inplace=True)
df['MinorAxisLength'] = MinorAxisLength

# Assuming the first column is the 'Class' column
y = df['Class']
X = df.iloc[:, :5]

# Split the data into training and testing sets for each class
train_indices = []
test_indices = []

# Iterate through each class
for class_label in y.unique():
    # Get indices for the current class
    class_indices = df[y == class_label].index

    # Split the indices into training (30 samples) and testing (20 samples)
    train_class_indices, test_class_indices = train_test_split(class_indices, test_size=20, random_state=42)

    # Add the indices to the overall lists
    train_indices.extend(train_class_indices[:30])  # Take the first 30 for training
    test_indices.extend(test_class_indices)

# Shuffle the indices
train_indices = np.random.permutation(train_indices)
test_indices = np.random.permutation(test_indices)

# Create training and testing sets
X_train = X.loc[train_indices].values
y_train = pd.get_dummies(y.loc[train_indices]).values

X_test = X.loc[test_indices].values
y_test = pd.get_dummies(y.loc[test_indices]).values
#################################################################################################

# User inputs
# hidden_layers = int(input("Enter number of hidden layers: "))
# neurons_per_layer = [int(input(f"Enter number of neurons in hidden layer {i + 1}: ")) for i in range(hidden_layers)]
# learning_rate = float(input("Enter learning rate (eta): "))
# epochs = int(input("Enter number of epochs (m): "))
# use_bias = input("Add bias? (y/n): ").lower() == 'y'
# activation_function = input("Choose activation function (sigmoid/tanh): ").lower()
neurons_per_layer = [3, 5]
learning_rate = 0.1
epochs = 10
use_bias = False
activation_function = "sigmoid"
# Assuming your data has 5 features and 3 classes
input_size = 5
output_size = 3


# Initialize and train the neural network
nn = NeuralNetwork(input_size, neurons_per_layer, output_size, learning_rate, epochs, use_bias, activation_function)
nn.train(X_train, y_train)

# Make predictions on the test set
predictions = nn.predict(X_test)

# Evaluate the performance
# print("np.argmax(y_test, axis=0) :", np.argmax(y_test, axis=0), "np.argmax(predictions, axis=0) :",
#       np.argmax(predictions, axis=0), sep="\n")
accuracy = accuracy_score(np.argmax(y_test, axis=0), np.argmax(predictions, axis=0))
# accuracy = accuracy_score(y_test, predictions)

print("Overall Accuracy:", accuracy)

# confusion_mat = confusion_matrix(np.argmax(y_test, axis=0), np.argmax(predictions, axis=0))
# print("Confusion Matrix:")
# print(confusion_mat)
# print(y_test, predictions, sep="\n")



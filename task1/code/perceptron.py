from skimage.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


def perceptron(eta, max_epochs, data, add_bias=True):
    """
    :param eta: learning rate
    :param data: the actual training data along with the actual value as last element
    :param add_bias: add a bias or not
    :return: the weights vector & the bias
    """
    

    w = np.random.rand(len(data[0])-1) # -1 to not inclue the last element which is the actual value


    b = 0
    if add_bias:
        b = np.random.rand()
    else:
        b = 0

    # Implement the single-layer perceptron algorithm
    for _ in range(int(max_epochs)):
        for i in range(len(data)):
            xi = data[i][:-1]
            ti = data[i][-1]

            xi = np.array(xi)

            # yi = (np.dot(w, xi) + b)


            yi = np.sign(np.dot(w, xi) + b) 
            yi = np.where(np.dot(w, xi) + b >= 0, 1, 0)
            if yi != ti:
                loss = ti - yi
                w += eta * loss * xi
                if b != 0:
                    b += eta * loss

            # loss = ti - yi
            # w += eta * loss * xi
            # if b != 0:
            #     b += eta * loss

    return w, b
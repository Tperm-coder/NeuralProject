import random
import numpy as np


class Adaline :
    learningRate = 0.001
    epochs = 100
    mseThresh = 0.001
    hasBias = True
    data = []

    def __init__(self, learningRate, epochs, mseThresh, hasBias, data):
        self.learningRate = learningRate
        self.epochs = epochs
        self.mseThresh = mseThresh
        self.hasBias = hasBias
        self.data = data
        print(self)

    def dotProduct(self, arr1, arr2):
        res = 0.0
        n = len(arr1)
        assert(n == len(arr2))

        for i in range(n):
            res += arr1[i] * arr2[i]
        
        return res

    def activation_function(self,x) :
        return x

    def fit(self, features = 2):

        # weightsVector = []

        weightsVector = np.random.rand(features + int(self.hasBias))
        # for i in range(features + int(self.hasBias)):
        #     weightsVector.append(random.random())


        mse = 0
        n = len(self.data)
        try :
            for _ in range(int(self.epochs)):
                res = 0
                mse = 0
                for i in range(n):

                    # row = self.data[i] // [v1,v2,actual]
                    # weight =              [bias,w1,w2]
                    if(self.hasBias):
                        res = np.dot(self.data[i][:-1], weightsVector[1:]) + weightsVector[0]
                    else:
                        res = np.dot(self.data[i][:-1], weightsVector)


                    # if (res + int(self.hasBias) >= 0):
                    #     res = 1
                    # else :
                    #     res = 0 

                    res = self.data[i][-1] - self.activation_function(res) 
                    # yi = np.where(np.dot(w, xi) + b >= 0, 1, 0)
                    print(self.data[i][-1])

                    if(self.hasBias):
                        weightsVector[0] += self.learningRate * res
                        weightsVector[1] += self.learningRate * res * self.data[i][0]
                        weightsVector[2] += self.learningRate * res * self.data[i][1]
                    else :
                        weightsVector[0] += self.learningRate * res * self.data[i][0]
                        weightsVector[1] += self.learningRate * res * self.data[i][1]


                    # for j in range(int(self.hasBias), len(weightsVector)):
                    #     weightsVector[j] += self.learningRate * res * self.data[i][j - int(self.hasBias)]

                print(weightsVector,"===" , res , '\n')


                res = 0 
                for i in range(n):
                    if(self.hasBias):
                        res = self.dotProduct(self.data[i][:-1], weightsVector[1:])
                    else:
                        res = self.dotProduct(self.data[i][:-1], weightsVector)


                
                
                res = self.data[i][-1] - res
                mse += (res**2) * 0.5
                mse /= n
                
                print(res,"===" , mse  , '---',self.mseThresh, '\n')

                if(mse <= self.mseThresh):
                    print("MSE reached")
                    break
        except :
            print("Overshoot")
            return []

        print("[][][][][][][][][=====]",weightsVector)
        return weightsVector



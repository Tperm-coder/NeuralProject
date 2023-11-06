import random
class Adaline :
    learningRate = 0.01
    epochs = 50
    mseThresh = 0.001
    hasBias = True
    data = []

    def __init__(self, learningRate, epochs, mseThresh, hasBias, data):
        self.learningRate = learningRate
        self.epochs = epochs
        self.mseThresh = mseThresh
        self.hasBias = hasBias
        self.data = data

    def dotProduct(self, arr1, arr2):
        res = 0.0
        n = len(arr1)
        assert(n == len(arr2))

        for i in range(n):
            res += arr1[i] * arr2[i]
        
        return res


    def fit(self, features = 2):

        weightsVector = []
        for i in range(features + int(self.hasBias)):
            weightsVector.append(random.random())


        mse = 0
        n = len(self.data)
        for _ in range(int(self.epochs)):
            res = 0
            mse = 0
            for i in range(n):
                print(int(self.hasBias),self.data[i],weightsVector)
                if(self.hasBias):
                    res = self.dotProduct(self.data[i][:-1], weightsVector[1:])
                else:
                    res = self.dotProduct(self.data[i][:-1], weightsVector)

                print("====>",res,"\n")
            
                if(self.hasBias):
                    weightsVector[0] += self.learningRate * res

                for j in range(int(self.hasBias), len(weightsVector)):
                    weightsVector[j] += self.learningRate * res * self.data[i][j - int(self.hasBias)]
            
                mse += res**2 * 0.5
            
            mse /= n
            if(mse <= self.mseThresh):
                break
        return weightsVector



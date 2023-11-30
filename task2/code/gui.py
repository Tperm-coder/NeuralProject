from PyQt5.QtWidgets import *
from PyQt5 import uic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from neural_network import NeuralNetwork

DATASET_PATH = "../Dataset/data.csv"


class CustomMinMaxScaler:
    def __init__(self):
        self.areaScaler = MinMaxScaler()
        self.perimeterScaler = MinMaxScaler()
        self.roundnessScaler = MinMaxScaler()
        self.majAlScaler = MinMaxScaler()
        self.minAlScaler = MinMaxScaler()

    def fit_transform(self, X):
        X["Area"] = self.areaScaler.fit_transform(X[["Area"]])
        X["Perimeter"] = self.perimeterScaler.fit_transform(X[["Perimeter"]])
        X["MajorAxisLength"] = self.majAlScaler.fit_transform(X[["MajorAxisLength"]])
        X["MinorAxisLength"] = self.minAlScaler.fit_transform(X[["MinorAxisLength"]])
        X["roundnes"] = self.roundnessScaler.fit_transform(X[["roundnes"]])
        return X


class MyGUI(QMainWindow):
    isModelTrained = False
    neuralClass = True
    areaScaler = True
    perimeterScaler = True
    roundnessScaler = True
    majAlScaler = True
    minAlScaler = True
    customMinMaxScaler = True


    def __init__(self,UI_file_path):
        super(MyGUI,self).__init__()
        uic.loadUi(UI_file_path,self)
        self.neuralClass = True
        

        self.show()

        self.trainButton.clicked.connect(self.onStartTrainingClick)
        self.predictBtn.clicked.connect(self.onPredict)

    def confusion_matrix(self,actual, predicted):
        
        print('----------------------')
        print(actual)
        print('----------------------')
        print(predicted)
    

        truePositive = 0
        trueNegative = 0

        falsePostive = 0
        falseNegative = 0

        for i in range(len(actual)) :
            if (actual[i] == predicted[i]) :
                if actual[i] == f1 :
                    truePositive += 1
                elif actual[i] == f2 :
                    trueNegative += 1
                continue

            if predicted[i] == f1 :
                falsePostive += 1
            elif predicted[i] == f2 :
                falseNegative += 1

        
        confusion_matrix_str = "          POS    |     NEG\n"
        confusion_matrix_str += "POS       " + str(truePositive) + "     " + str(falsePostive) + "\n"
        confusion_matrix_str += "NEG       " + str(falseNegative) + "     " + str(trueNegative) + "\n"

        return confusion_matrix_str

    
    def showPopUp(self,title,msgTxt) :
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(str(msgTxt))
        msg.setWindowTitle(str(title))
        result = msg.exec_()


    def onStartTrainingClick(self) :

        try :

            networkStructure = str(self.networkStructure.toPlainText()).split(',')
            for i in range(len(networkStructure)) :
                networkStructure[i] = int(networkStructure[i])
            epochs = int(self.epochsInp.text())
            learningRate = float(self.learningInp.text())
            isBiasAdded = bool(self.isBiased.isChecked())
            activationFunction = "sigmoid"
            
            if (bool(self.isHyper.isChecked())) :
                activationFunction = "tanh"

            print(networkStructure,epochs,learningRate,isBiasAdded,activationFunction)

            # data preprocessing & splitting
            data = pd.read_csv(DATASET_PATH)
            df = pd.DataFrame(data)

            MinorAxisLength = df['MinorAxisLength']
            ave = MinorAxisLength.mean()
            MinorAxisLength.fillna(ave, inplace=True)
            df['MinorAxisLength'] = MinorAxisLength
            # Assuming the first column is the 'Class' column
            y = df['Class']
            X = df.iloc[:, :5]
            # X = pd.DataFrame(X)

            self.customMinMaxScaler = CustomMinMaxScaler()
            X = self.customMinMaxScaler.fit_transform(X.copy())
            print(X)
            X = pd.DataFrame(X)

            # Split the data into training and testing sets for each class
            test_indices = []
            train_indices = []

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


            # Initialize and train the neural network
            self.neuralClass = NeuralNetwork(5, networkStructure, 3, learningRate, epochs, False, activationFunction)
            self.neuralClass.train(X_train, y_train)

            # Make predictions on the test set
            predictions = self.neuralClass.predict(X_test)

            # Evaluate the performance
            # print("np.argmax(y_test, axis=0) :", np.argmax(y_test, axis=0), "np.argmax(predictions, axis=0) :",
            #       np.argmax(predictions, axis=0), sep="\n")
            # print(y_test)
            # print('-------------------')
            # print(predictions)

            cnt = 0
            _pred = []
            _actual = []
            for predI in range(len(predictions)):
                testClass = -1
                predictedClass = -1
                for i in range(len(predictions[predI])):
                    if(predictions[predI][i] > 0.0):
                        predictedClass = i
                _pred.append(predictedClass)

                for i in range(len(y_test[predI])):
                    if(y_test[predI][i] > 0.0):
                        testClass = i
                _actual.append(testClass)

                if(testClass == predictedClass):
                    cnt += 1

            confusion_mat = confusion_matrix(_pred ,_actual)
            print("Confusion Matrix:")
            print(confusion_mat)

            predictions = self.neuralClass.predict(X_train)

            __cnt = 0
            __pred = []
            __actual = []
            for predI in range(len(predictions)):
                testClass = -1
                predictedClass = -1
                for i in range(len(predictions[predI])):
                    if(predictions[predI][i] > 0.0):
                        predictedClass = i
                __pred.append(predictedClass)

                for i in range(len(y_train[predI])):
                    if(y_train[predI][i] > 0.0):
                        testClass = i
                __actual.append(testClass)

                if(testClass == predictedClass):
                    __cnt += 1

            acc = ("Test Accuracy:\n" +  str(cnt/len(y_test)*100)+' %\n\n'+ "Train Accuracy:\n" +  str(__cnt/len(y_train)*100)+' %'+      "\n\nConfusion Matrix :\n"+str(confusion_mat))
            self.showPopUp("Training is done",acc)


            self.isModelTrained = True
        except :
            self.showPopUp("Error" , "A runtime error occurred")

    def onPredict(self) :
        if (not self.isModelTrained) :
            self.showPopUp("Warning", "You need to train the model first")
            return

        area = float(self.area.text())
        perimeter = float(self.perimeter.text())
        major = float(self.majoral.text())
        minor = float(self.minoral.text())
        roundness = float(self.roundness.text())

        X = {
            'Area': [area],
            'Perimeter': [perimeter],
            'MajorAxisLength': [major],
            'MinorAxisLength': [minor],
            'roundnes': [roundness],
        }
        X = pd.DataFrame(X)

        X = self.customMinMaxScaler.fit_transform(X.copy())
        X = pd.DataFrame(X)

        preds = self.neuralClass.predict(X.values)
        print("Predicted" ,preds)

        predictedClass = -1
        for i in range(len(preds[0])) :
            if (preds[0][i] == 1) :
                predictedClass = i

        _class = 0
        if (predictedClass == 0) :
            _class = "BOMBAY"
        if (predictedClass == 1) :
            _class = "CALI"
        if (predictedClass == 2) :
            _class = "SIRA"
        
        self.showPopUp("Done", "The predicted class is:\n" + _class)


        


from PyQt5.QtWidgets import *
from PyQt5 import uic
from adaline import Adaline
from perceptron import perceptron
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



DATASET_PATH = "../Dataset/data.csv"

class MyGUI(QMainWindow):

    # global store acting as a single source of truth to share state across the app
    # and to execute global functionalities
    store = dict({})

    def __init__(self,UI_file_path):
        super(MyGUI,self).__init__()
        uic.loadUi(UI_file_path,self)

        
        self.store = dict({
            # get input fields' values
            "get" : {
                "learningRate" : lambda : float(self.learningRateSpinBox.text()),
                "epocs" : lambda : float(self.epocsSpinBox.text()),
                "mseThreshold" : lambda : float(self.epocsSpinBox.text()),
                "isAddBias" : lambda : bool(self.addBiasCheckBox.isChecked()),
                "isAdeline": lambda : bool(self.adalineRadioButton.isChecked()),
                "isPercetron": lambda : bool(self.perceptronRadioButton.isChecked()),

                # classification classes initials
                "isBSclasses": lambda : bool(self.BSradioButton.isChecked()),
                "isCSclasses": lambda : bool(self.CSradioButton.isChecked()),
                "isBCclasses": lambda : bool(self.BCradioButton.isChecked()),

                # features
                "Area": lambda : bool(self.area.isChecked()),
                "MajorAxisLength": lambda : bool(self.major.isChecked()),
                "MinorAxisLength": lambda : bool(self.minor.isChecked()),
                "roundnes": lambda : bool(self.round.isChecked()),
                "Perimeter": lambda : bool(self.permeter.isChecked())
            }
            
        }) 
        self.Adaline = {}

        #init functions
        self.show()

        #----------------------EVENT HANDLERES---------------------------
        #--------------Buttons
        self.startButton.clicked.connect(self.onStartButtonClick)

    # --------------------- Store Getters
    def getEpocs(self) :
        return self.store["get"]["epocs"]()

    def getLearningRate(self) :
        return self.store["get"]["learningRate"]()
    
    def getMSEThreshold(self) :
        return self.store["get"]["mseThreshold"]()

    def getMSEThreshold(self) :
        return self.store["get"]["mseThreshold"]()

    def getIsAdaline(self) :
        return self.store["get"]["isAdeline"]()

    def getIsBSclasses(self) :
        return self.store["get"]["isBSclasses"]()
    
    def getIsCSclasses(self) :
        return self.store["get"]["isCSclasses"]()

    def getIsBCclasses(self) :
        return self.store["get"]["isBCclasses"]()
    
    def getIsBiased(self) :
        return self.store["get"]["isBCclasses"]()

    def getIsAddBias(self) :
        return self.store["get"]["isAddBias"]()

    def getChoosenFeatures(self) :
        choosenFeauters = []
        for f in ["Area" , "MajorAxisLength" , "MinorAxisLength" , "roundnes" , "Perimeter"] : 
            if (self.store["get"][f]()) :
                choosenFeauters.append(f)

        return choosenFeauters
    #---------------------------------------



    #---------------Helper functions
    def _showPopUp(self,title,msgTxt) :
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(str(msgTxt))
        msg.setWindowTitle(str(title))
        result = msg.exec_()


    def getTrainingDataAfterFormating(self,choice,features,normalize = False):
        data = pd.read_csv(DATASET_PATH)

        data = data.fillna(data.mean())

        class_mapping = {"BOMBAY": 0, "CALI": 1, "SIRA": 2}
        data["Class"] = data["Class"].map(class_mapping)

        _features = features.copy()
        if normalize :
            scaler = MinMaxScaler()
            data[_features] = scaler.fit_transform(data[_features])

        _features.append("Class")
        selected_columns = data[_features]

        data_as_list_of_lists = selected_columns.values.tolist()


        data = []
        for row in data_as_list_of_lists :
            if (choice == 0.0) :
                if (row[-1] == 0.0 or row[-1] == 2.0) :
                    data.append(row)
            elif (choice == 1.0) :
                if (row[-1] == 1.0 or row[-1] == 2.0) :
                    data.append(row)
            elif (choice == 2.0) :
                if (row[-1] == 1.0 or row[-1] == 0.0) :
                    data.append(row)
        
        return data

    def spilitData(self,data) :
        c1 = []
        c2 = []

        for i in range(len(data)) :
            if (data[i][-1] == data[0][-1]) :
                c1.append(data[i])
            else :
                c2.append(data[i])

        x = 30
        c1 = np.array(c1)
        c1_train,c1_test  = train_test_split(c1, test_size=x, shuffle=True)
        c2_train,c2_test  = train_test_split(c2, test_size=x, shuffle=True)

        print(len(c1_train),len(c1_test))


        
        return c1_train,c1_test,c2_test,c2_test

    def drawGraph(self,data,weights,b) :
        x = []
        y = []
        colors = []

        _min = 9999999999999
        _max = -1

        for i in range(len(data)):
            _max = max(_max,max(data[i][:-1]))
            _min = min(_min,max(data[i][:-1]))
            
            x.append(data[i][0])
            y.append(data[i][1])

            if (data[i][-1] == data[0][-1]) :
                colors.append("blue")
            else :
                colors.append("black")

        w1 = weights[0]
        w2 = weights[1]


        x = np.array(x)
        y = np.array(y)

        plt.scatter(x, y, c=colors, marker='o', label='Training Data Points')

        x_boundary = np.linspace(_min, _max, 10000)
        y_boundary = (-w1 * x_boundary - b) / w2

        plt.plot(x_boundary, y_boundary, color='red', label='Decision Boundary')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True)
        plt.show()

    def confusion_matrix(self,actual, predicted):

        f1 = list(set(actual))[0] # positive
        f2 = list(set(actual))[1] # negative

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
            




    #-------------------------------

    def onStartButtonClick(self) :

        # get input fields values
        learningRate = self.getLearningRate()
        epocs = self.getEpocs()
        mseThreshold = self.getMSEThreshold()
        isAdeline = self.getIsAdaline()
        isAddBias = self.getIsAddBias()
        isBS = self.getIsBSclasses()
        isCS = self.getIsCSclasses()
        isBC = self.getIsBCclasses()

        
        choice = -1
        if (isBS) :
            choice = 0
        elif (isCS) :
            choice = 1
        else :
            choice = 2

        features = self.getChoosenFeatures()

        if (len(features) != 2): 
            self._showPopUp("Warning","The number of features must be equal to 2")
            return
        
        data = self.getTrainingDataAfterFormating(choice,features,True)
        c1_train,c1_test,c2_train,c2_test = self.spilitData(data)

        data = []
        test = []

        for i in range(len(c1_train)) :
            data.append(c1_train[i])
            data.append(c2_train[i])

        for i in range(len(c1_test)) :
            test.append(c1_test[i])
            test.append(c2_test[i])

        weights = []
        bias = 0

        if isAdeline :
            print("Using adaline")
            self.Adaline = Adaline(learningRate,epocs,mseThreshold,isAddBias,data)
            weightsVector = self.Adaline.fit(len(features))

            if weightsVector == False :
                self._showPopUp("Error" , "Gradient decent overshooted\n try smaller learning rate")
                return
            if (isAddBias) :
                bias = weightsVector[0]
                weights = weightsVector[1:]
            else :
                bias = 0
                weights = weightsVector

        else :
            print("Using perceptron")
            weights,bias = perceptron(learningRate,epocs,data,isAddBias)

        print(weights,bias)
        y_pred = np.zeros(len(test))
        for i in range(len(test)):
            y_pred[i] = np.where(np.dot(weights, test[i][:-1]) + bias >= 0, 1, 0) # segum like activation function

        y_test = []
        for i in test :
            y_test.append(i[-1])

        accuracy = accuracy_score(y_test, y_pred)
        accuracy *= 100


        conf_mat = self.confusion_matrix(y_test,y_pred)
        print(conf_mat)
        self._showPopUp("Results" , "Accuracy\n" + str(accuracy) + "%\n\n" + conf_mat)

        self.drawGraph(data,weights,bias)
        return
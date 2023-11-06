from PyQt5.QtWidgets import *
from PyQt5 import uic
from adaline import Adaline
import pandas as pd

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


    def getTrainingDataAfterFormating(self,choice,features):
        data = pd.read_csv(DATASET_PATH)

        data = data.fillna(data.mean())

        class_mapping = {"BOMBAY": 0, "CALI": 1, "SIRA": 2}
        data["Class"] = data["Class"].map(class_mapping)

        _features = features.copy()
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

    #-------------------------------
    def onStartButtonClick(self) :

        # get input fields values
        learningRate = self.getLearningRate()
        epocs = self.getEpocs()
        mseThreshold = self.getMSEThreshold()
        isAdeline = self.getIsAdaline()
        isPercetron = not isAdeline
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
        
    
        data = self.getTrainingDataAfterFormating(choice,features)
        self.Adaline = Adaline(learningRate,epocs,mseThreshold,isAddBias,data)
        print(self.Adaline.fit(len(features))) # -1 for the class column

    
        return